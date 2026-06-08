from time import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from sklearn.cluster import KMeans
import metrics
from OGConvAE import CAE


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim),
                                        initializer='glorot_uniform', trainable=True)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters),
                                axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / (K.sum(q, axis=1) + 1e-10))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphRegLayer(Layer):
    def __init__(self, lambda_g=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.lambda_g = lambda_g
        self.graph_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=None):
        Z, Q = inputs
        batch_size = tf.cast(tf.shape(Q)[0], tf.float32)

        z_norm = tf.reduce_sum(tf.square(Z), axis=1, keepdims=True)
        dist = z_norm + tf.transpose(z_norm) - 2.0 * tf.matmul(Z, Z, transpose_b=True)

        dist = tf.clip_by_value(dist, 0.0, 100.0)
        sigma = tf.reduce_mean(dist) + 1e-8
        S = tf.exp(-dist / (2.0 * sigma))

        D = tf.reduce_sum(S, axis=1)
        D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(D + 1e-8))
        L = tf.eye(tf.shape(S)[0]) - tf.matmul(tf.matmul(D_inv_sqrt, S), D_inv_sqrt)

        graph_loss = self.lambda_g * tf.linalg.trace(tf.matmul(tf.matmul(Q, L, transpose_a=True), Q))
        graph_loss = graph_loss / batch_size

        self.graph_loss.assign(graph_loss)
        self.add_loss(graph_loss)

        return Q


def kld_loss(gamma=0.1):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        y_true = tf.clip_by_value(y_true, 1e-8, 1.0)
        return gamma * tf.reduce_mean(
            tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=1)
        )
    return loss


class DCEC(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0,
                 lambda_g=0.0001):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.lambda_g = lambda_g
        self.pretrained = False
        self.y_pred = []
        self.best_metrics = {'acc': 0.0, 'nmi': 0.0, 'ari': 0.0, 'f1': 0.0}

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name='embedding').output
        Q = ClusteringLayer(self.n_clusters, name='clustering')(hidden)

        self.graph_reg_layer = None
        if lambda_g > 0:
            self.graph_reg_layer = GraphRegLayer(lambda_g=lambda_g, name='graph_reg')
            Q = self.graph_reg_layer([hidden, Q])
        self.model = Model(inputs=self.cae.inputs, outputs=[Q, self.cae.outputs[0]])
        self.encoder = Model(inputs=self.cae.inputs, outputs=hidden)


    def pretrain(self, x, batch_size=256, epochs=50, save_dir='results/temp'):
        print('...Pretraining...')
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.cae.compile(optimizer=optimizer, loss='mse')
        from tensorflow.keras.callbacks import CSVLogger
        csv_logger = CSVLogger(save_dir + '/pretrain_log.csv')

        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        print('Pretraining time: ', time() - t0)
        weights_path = save_dir + '/pretrain_cae.weights.h5'
        self.cae.save_weights(weights_path)
        print('Pretrained weights are saved to %s' % weights_path)
        self.pretrained = True


    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)  # Return the index of the maximum value in the prediction result q, that is, the prediction result

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / (q.sum(0) + 1e-10)
        return (weight.T / (weight.sum(1) + 1e-10)).T

    def compile(self, gamma=0.1, optimizer='adam'):
        self.model.compile(
            optimizer=optimizer,
            loss=[kld_loss(gamma), 'mse'],
            loss_weights=[1.0, 1.0]
        )

    def fit(self, x, y=None, batch_size=256, maxiter=30000, tol=0.001,
            update_interval=140, cae_weights=None, save_dir='./results/temp', epochs=50):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        best_acc = 0.0
        best_nmi = 0.0
        best_ari = 0.0
        best_f1 = 0.0

        t0 = time()
        import os
        auto_weights_path = os.path.join(save_dir, 'pretrain_cae.weights.h5')

        if cae_weights is None and os.path.exists(auto_weights_path):
            print(f'Pre-trained weights are already available: {auto_weights_path}')
            cae_weights = auto_weights_path
        if cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print(f'Successfully loaded pre-trained weights: {cae_weights}')
            self.pretrained = True
        elif not self.pretrained:
            # If no weights are available, perform pre-training
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=%d' % epochs)
            self.pretrain(x, batch_size, epochs=epochs, save_dir=save_dir)

        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])


        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, clipnorm=1.0)
        self.compile(gamma=0.1, optimizer=optimizer)
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/training_history.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'f1',
                                                        'best_acc', 'best_nmi', 'best_ari', 'best_f1',
                                                        'Loss', 'kl_loss', 'recon_loss', 'graph_loss'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0, 0]
        index = 0

        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    f1 = np.round(metrics.f1(y, self.y_pred), 5)

                    loss_rounded = np.round(loss, 5)
                    Loss = loss_rounded[0]
                    kl_loss_val = loss_rounded[1]
                    recon_loss_val = loss_rounded[2]
                    graph_loss_val = loss_rounded[3]

                    if acc > best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
                        self.best_metrics = {'acc': best_acc, 'nmi': best_nmi, 'ari': best_ari, 'f1': best_f1, }

                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, f1=f1,
                                   best_acc=best_acc, best_nmi=best_nmi, best_ari=best_ari, best_f1=best_f1,
                                   Loss=Loss, kl_loss=kl_loss_val, recon_loss=recon_loss_val, graph_loss=graph_loss_val)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('Iter', ite, ': Acc=', acc, ', nmi=', nmi, ', ari=', ari, ', F1=', f1,
                          '; Loss=', Loss, '; KL=', kl_loss_val, '; Recon=', recon_loss_val, '; Graph=', graph_loss_val)

                # stopping condition
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol, 'Training terminated！')
                    self.model.save_weights(save_dir + '/dcec_model_best.weights.h5')
                    logfile.close()
                    break

            if (index + 1) * batch_size > x.shape[0]:
                original_loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                          y=[p[index * batch_size::], x[index * batch_size::]])
                graph_loss_val = self.graph_reg_layer.graph_loss.numpy() if (
                        self.lambda_g > 0 and self.graph_reg_layer is not None) else 0.0
                loss = [
                    original_loss[0],  # total loss
                    original_loss[1],  # KL loss
                    original_loss[2],  # reconstruction loss
                    graph_loss_val     # graph loss
                ]
                index = 0
            else:
                original_loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                          y=[p[index * batch_size:(index + 1) * batch_size],
                                                             x[index * batch_size:(index + 1) * batch_size]])
                graph_loss_val = self.graph_reg_layer.graph_loss.numpy() if (
                        self.lambda_g > 0 and self.graph_reg_layer is not None) else 0.0
                loss = [original_loss[0], original_loss[1], original_loss[2], graph_loss_val]
                index += 1

            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.weights.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.weights.h5')

            ite += 1

        #save model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.weights.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.weights.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)


if __name__ == "__main__":

    tf.keras.backend.clear_session()
    import argparse

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='fashion_mnist',
                        choices=['mnist', 'usps', 'mnist-test', 'fashion_mnist'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=30000, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--lambda_g', default=0.0001, type=float,
                        help='coefficient of graph regularization')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs for pretraining')
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f'results/{args.dataset}'

    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(args)

    #load data
    from datasets import load_mnist, load_usps, load_fashion_mnist

    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
    elif args.dataset == 'mnist-test':
        x, y = load_mnist()
        x, y = x[60000:], y[60000:]
    elif args.dataset == 'fashion_mnist':
        x, y = load_fashion_mnist()

    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],
                n_clusters=args.n_clusters, lambda_g=args.lambda_g)
    dcec.model.summary()

    optimizer = 'adam'
    dcec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             cae_weights=args.cae_weights,
             epochs=args.epochs)
    print('\n========================================')
    best_acc = dcec.best_metrics['acc']
    best_nmi = dcec.best_metrics['nmi']
    best_ari = dcec.best_metrics['ari']
    best_f1 = dcec.best_metrics['f1']
    print(f'acc = {best_acc:.4f}, nmi = {best_nmi:.4f}, ari = {best_ari:.4f}, f1 = {best_f1:.4f}')