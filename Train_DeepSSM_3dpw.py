import tensorflow as tf
import os.path
import numpy as np
from nets import DeepSSM_3dpw as  DeepSSM
from data_provider import datasets_factory_joints_3dpw as datasets_factory
from utils import metrics
from utils import optimizer
import time
import scipy.io as io
import os,shutil
import pdb

FLAGS = tf.app.flags.FLAGS
# data path
tf.app.flags.DEFINE_string('dataset_name', 'skeleton',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/moving-mnist-example/moving-mnist-train.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('test_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'test data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/mnist_predcnn',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_dir', 'results/mnist_predcnn',
                           'path to save generate results')
tf.app.flags.DEFINE_string('bak_dir', 'results/mnist_predcnn/bak',
                            'dir to backup result.')
# model parameter
tf.app.flags.DEFINE_string('pretrained_model','',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 35,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('joints_number', 23,
                            'the number of joints of a pose')
tf.app.flags.DEFINE_integer('joint_dims', 3,
                            'one joints dims.')

tf.app.flags.DEFINE_integer('stacklength1', 2,
                            'stack trajblock number.')
tf.app.flags.DEFINE_integer('stacklength2', 1,
                            'stack trajblock number.')		

tf.app.flags.DEFINE_integer('filter_size', 3,
                            'filter size.')

# opt
tf.app.flags.DEFINE_float('lr', 0.0001,
                          'base learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 100000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 20,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')
tf.app.flags.DEFINE_integer('num_save_samples', 100000,
                            'number of sequences to be saved.')
tf.app.flags.DEFINE_integer('n_gpu', 4,
                            'how many GPUs to distribute the training across.')
#num_hidden=[64,64,64,64,64]
num_hidden=64
print'!!! Network:', num_hidden
class Model(object):
	def __init__(self):
		# inputs
		self.x = [tf.placeholder(tf.float32,[FLAGS.batch_size,
											FLAGS.seq_length,
											FLAGS.joints_number,
											FLAGS.joint_dims])
				for i in range(FLAGS.n_gpu)]
		grads = []
		loss_train = []
		self.pred_seq = []
		self.tf_lr = tf.placeholder(tf.float32, shape=[])
		self.keep_prob = tf.placeholder(tf.float32)
		self.params = dict()
        

		for i in range(FLAGS.n_gpu):
			with tf.device('/gpu:%d' % i):
				with tf.variable_scope(tf.get_variable_scope(),
						reuse=True if i > 0 else None):
					# define a model
					output_list = DeepSSM.DeepSSM(
							self.x[i],
							self.keep_prob,
							FLAGS.seq_length,
							FLAGS.input_length,
							FLAGS.stacklength1,
							FLAGS.stacklength2,
							num_hidden,
							FLAGS.filter_size)

					gen_ims = output_list[0]
					loss = output_list[1]
					pred_ims = gen_ims[:, FLAGS.input_length - FLAGS.seq_length:]
					loss_train.append(loss)   ###
					# gradients
					all_params = tf.trainable_variables()
					grads.append(tf.gradients(loss, all_params))
					self.pred_seq.append(pred_ims)

		if FLAGS.n_gpu == 1:
			self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
		else:
			# add losses and gradients together and get training updates
			with tf.device('/gpu:0'):
				for i in range(1, FLAGS.n_gpu):
					loss_train[0] += loss_train[i]
					for j in range(len(grads[0])):
						grads[0][j] += grads[i][j]
			# keep track of moving average
			ema = tf.train.ExponentialMovingAverage(decay=0.9995)
			maintain_averages_op = tf.group(ema.apply(all_params))
			self.train_op = tf.group(optimizer.adam_updates(
				all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
				maintain_averages_op)

		self.loss_train = loss_train[0] / FLAGS.n_gpu

		# session
		variables = tf.global_variables()
		self.saver = tf.train.Saver(variables)
		init = tf.global_variables_initializer()
		configProt = tf.ConfigProto()
		configProt.gpu_options.allow_growth = True
		configProt.allow_soft_placement = True
		self.sess = tf.Session(config = configProt)
		self.sess.run(init)
		if FLAGS.pretrained_model:
			print 'pretrain model: ',FLAGS.pretrained_model
			self.saver.restore(self.sess, FLAGS.pretrained_model)

	def train(self, inputs, lr, keep_prob):
		feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
		feed_dict.update({self.tf_lr: lr})
		feed_dict.update({self.keep_prob: keep_prob})
		loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
		return loss

	def test(self, inputs, keep_prob):
		feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
		feed_dict.update({self.keep_prob: keep_prob})
		gen_ims = self.sess.run(self.pred_seq, feed_dict)
		return gen_ims

	def save(self, itr):
		checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
		self.saver.save(self.sess, checkpoint_path, global_step=itr)
		print('saved to ' + FLAGS.save_dir)


def main(argv=None):
	if not tf.gfile.Exists(FLAGS.save_dir):
		tf.gfile.MakeDirs(FLAGS.save_dir)
	if not tf.gfile.Exists(FLAGS.gen_dir):
		tf.gfile.MakeDirs(FLAGS.gen_dir)

	print 'start training !',time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time()))
	# load data
	train_input_handle, test_input_handle = datasets_factory.data_provider(
		FLAGS.dataset_name,FLAGS.train_data_paths, FLAGS.valid_data_paths,
		FLAGS.batch_size * FLAGS.n_gpu, FLAGS.joints_number,FLAGS.input_length,FLAGS.seq_length,is_training=True)

	print('Initializing models')
	model = Model()
	lr = FLAGS.lr
	train_time=0
	test_time_all=0
	folder=1
	path_bak=FLAGS.bak_dir

	test_interval=FLAGS.test_interval
	snapshot_interval=FLAGS.snapshot_interval
	min_err=1000.0
	err_list=[]
	Keep_prob = 0.75
	for itr in range(1, FLAGS.max_iterations + 1):
		if train_input_handle.no_batch_left():
			train_input_handle.begin(do_shuffle=True)
		'''
		if itr % 2000 == 0:
			lr = lr* 0.95
		'''
		start_time = time.time()
		ims = train_input_handle.get_batch()
		ims = ims[:,:,0:FLAGS.joints_number,:]
		ims_list = np.split(ims, FLAGS.n_gpu)
		cost = model.train(ims_list, lr, Keep_prob)
		# inverse the input sequence
		ims1=ims[:, ::-1]
		ims_list1 = np.split(ims1, FLAGS.n_gpu)
		cost += model.train(ims_list, lr, Keep_prob)
		cost = cost/2
		end_time = time.time()
		t = end_time-start_time
		train_time += t
		if itr>150000:
			test_interval = 20
			snapshot_interval = 20
		if itr % FLAGS.display_interval == 0:
			print('itr: ' + str(itr)+' lr: '+str(lr)+' training loss: ' + str(cost))

		if itr % test_interval == 0:
			print('train time:'+ str(train_time))
			print('test...')

			test_input_handle.begin(do_shuffle=False)		

			res_path = os.path.join(FLAGS.gen_dir, str(itr))
			if  not tf.gfile.Exists(res_path):
				os.mkdir(res_path)
			avg_mse = 0
			batch_id = 0
			test_time=0
			mpjpe = np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
			f = 0
			while(test_input_handle.no_batch_left() == False):
				start_time1 = time.time()
				batch_id = batch_id + 1
				mpjpe1=np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
				test_ims = test_input_handle.get_batch()
				test_ims1 = test_ims
				test_ims=test_ims[:,:,0:FLAGS.joints_number,:]
				
				test_dat=test_ims[:,0:FLAGS.seq_length,:,:]
				gt_frm = test_ims1[:,FLAGS.input_length:]	

				test_dat = np.split(test_dat, FLAGS.n_gpu)
				img_gen = model.test(test_dat,1.0)
				end_time1 = time.time()
				t1=end_time1-start_time1
				test_time += t1
				# concat outputs of different gpus along batch
				img_gen = np.concatenate(img_gen)
				tem = gt_frm[:,:,-1]
				tem = np.expand_dims(tem,axis=2)
				img_gen=np.concatenate((img_gen,tem),axis=2)
				# mpjpe1=np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
				# MSE per frame
				for i in range(FLAGS.seq_length - FLAGS.input_length):
					x = gt_frm[:, i , :, ]
					gx = img_gen[:, i, :, ]
					for j in range(FLAGS.batch_size * FLAGS.n_gpu):
						tem1=0
						for k in range(gt_frm.shape[2]):
							tem1 += np.sqrt(np.square(x[j,k] - gx[j,k]).sum())
						mpjpe1[0,i] += tem1/(gt_frm.shape[2])
		
				# save prediction examples
				path = os.path.join(res_path, str(batch_id))
				if  not tf.gfile.Exists(path):
					os.mkdir(path)
				mpjpe1 = mpjpe1/(FLAGS.batch_size * FLAGS.n_gpu)
				mpjpe +=mpjpe1
				
				test_input_handle.next()
			test_time_all += test_time
			mpjpe=mpjpe/(batch_id)*1000
			err_list.append(np.mean(mpjpe))
			print( 'mean per joints position error: '+str(np.mean(mpjpe)))
			for i in range(FLAGS.seq_length - FLAGS.input_length):
				print(mpjpe[0,i])
			print 'current test time:'+str(test_time)
			print 'all test time: '+str(test_time_all)
			filename = os.path.join(res_path, 'test_result')
			io.savemat(filename, {'mpjpe':mpjpe})


		if itr % snapshot_interval == 0 and min(err_list) < min_err:
			model.save(itr)
			min_err = min(err_list)
			print 'model saving done! ', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time()))

		if itr % snapshot_interval == 0:
			print 'current minimize error is: ', min_err

		train_input_handle.next()

if __name__ == '__main__':
	tf.app.run()










