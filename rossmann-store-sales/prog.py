import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import math
import cPickle as pickle
import random

from sklearn import decomposition
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model

import xgboost as xgb

store = []
train = []
test = []

### Functions
def ohe(data, fit, transform, params, cols=None):
	if cols==None:
		cols = np.arange(data.shape[1])
	
	if fit:
		_feature_vals = [[]]*data.shape[1]
		for i in cols:
			vals = data[:,i]
			unique_vals = np.unique(vals)
			print unique_vals
			_feature_vals[i] = {unique_vals[i]: i for i in range(len(unique_vals))}
			for j in range(data.shape[0]):
				data[j,i] = _feature_vals[i][data[j,i]]
			# print len(feature_vals[i])
		_ohe = preprocessing.OneHotEncoder(n_values='auto', categorical_features='all', dtype=np.int, sparse=True, handle_unknown='error')
		print "OHE Expansion: "+str([len(x) for x in _feature_vals])
		del params[:]
		params.append(_feature_vals)
		params.append(_ohe)
		
		return data
	
	if transform:
		if not fit:
			_feature_vals = params[0]
			_ohe = params[1]
			for i in cols:
				for j in range(data.shape[0]):
					data[j,i] = _feature_vals[i][data[j,i]]
	
		if len(cols)==data.shape[1]:
			return _ohe.fit_transform(data)
		else:
			_cols = [x for x in range(data.shape[1]) if x not in cols]
			return sparse.hstack([_ohe.fit_transform(data[:,cols]), sparse.csr_matrix(data[:,_cols].astype(float))])

def dateSplit(date):
	# FORMAT: YYYY-MM-DD
	return date.split('-')

### Evaluation
def evalScore(y, y_pred):
	return math.sqrt(sum(((y[y!=0]-y_pred[y!=0])/y[y!=0])*((y[y!=0]-y_pred[y!=0])/y[y!=0]))/len(y[y!=0]))

def ToWeight(y):
	w = np.zeros(y.shape, dtype=float)
	ind = y != 0
	w[ind] = 1./(y[ind]**2)
	return w

def evalScore_xg(y_pred, y):
	# y = y.values
	y = np.array(y.get_label()).astype(float)
	y_pred = np.array(y_pred).astype(float)
	return math.sqrt(sum(((y[y!=0]-y_pred[y!=0])/y[y!=0])*((y[y!=0]-y_pred[y!=0])/y[y!=0]))/len(y[y!=0]))

### Load Data
def read(filename):
	object = []
	with open(filename, 'rb') as data:
		csvReader = csv.reader(data)
		for row in csvReader:
			object.append(row)
	
	return np.array(object)

store = read("store.csv")
train = read("train.csv")
test = read("test.csv")
###

###
store = store[1:,:]
train = train[1:,:]
test = test[1:,:]
###

### COARSE TUNING
def split_col(data, col):
	mod_data = []
	data_sorted = sorted(zip(np.arange(len(data)),data), key=lambda x:x[1][col])
	shuffle_idx, data_sorted = zip(*data_sorted)
	data_sorted = np.array(data_sorted)
	slices = [0]
	prev = data_sorted[0][col]
	for i, x in enumerate(data_sorted):
		if prev != x[col]:
			prev = x[col]
			slices.append(i)
	# plt.figure(1)
	# plt.subplot(211)
	# plt.plot(data_sorted[:,-2].astype(float))
	# plt.subplot(212)
	# unq = np.unique(data_sorted[:,col]).tolist()
	# plt.plot([unq.index(x) for x in data_sorted[:,col]])
	# plt.show()
	slices.append(len(data_sorted))
	mod_data = [data_sorted[slices[i]:slices[i+1]] for i in range(len(slices)-1)]
	return np.array(mod_data), np.array(shuffle_idx)

def coarseModel(X, y, fit, transform, params):
	if fit:
		train_store, _ = split_col(np.c_[X, y], 0)
		ovrl_avg = [None]*len(train_store)
		week_avg = [None]*len(train_store)
		sthldy_avg = [None]*len(train_store)
		schldy_avg = [None]*len(train_store)
		_stds = []
		_scores = []
		for i, _store in enumerate(train_store):
			_i = i
			_store = np.array(_store)
			_store = np.c_[_store, np.arange(len(_store))]
			_store_num = int(_store[0][0])
			i = _store_num-1
			ovrl_avg[i] = {}
			week_avg[i] = {}
			sthldy_avg[i] = {}
			schldy_avg[i] = {}
			print "Store: "+str(_store_num)+"..."
			# mean = np.mean(_store[:,-2].astype(float))
			# std = np.std(_store[:,-2].astype(float))
			# print std
			# plt.plot(_store[:,-2].astype(float))
			# plt.suptitle(str(_store_num)+" | ORIGINAL"+" | MEAN: {0:.4f}".format(mean)+" | STD: {0:.4f}".format(std))
			# plt.show()
			
			## OVRL
			_split, _ = split_col(_store, 1)
			for x in _split:
				x = np.array(x)
				ovrl_avg[i][x[0,1]] = np.mean(np.array(x[:,-2].astype(float)))
				for _x in x:
					_store[_x[-1],-2] = float(_store[_x[-1],-2]) - ovrl_avg[i][x[0,1]]
			# mean = np.mean(_store[:,-2].astype(float))
			# std = np.std(_store[:,-2].astype(float))
			# print std
			# plt.plot(_store[:,-2].astype(float))
			# plt.suptitle(str(_store_num)+" | After COMP"+" | MEAN: {0:.4f}".format(mean)+" | STD: {0:.4f}".format(std))
			# plt.show()
			# found = 0
			# 
			# if store[_store_num-1,1] != '' and int(store[_store_num-1,5]) > 2012:
			#	   if int(store[_store_num-1,4]) < 10:
			#			   date = str(store[_store_num-1,5]) + "-0" + str(int(store[_store_num-1,5]))
			#	   else:
			#			   date = str(store[_store_num-1,5]) + "-" + str(int(store[_store_num-1,5]))
			#	   for j, x in enumerate(_store):
			#			   if date > x[2]:
			#					   ovrl_avg[i]['<'] = np.mean(np.array(_store[:j,-2]).astype(float))
			#					   ovrl_avg[i]['>'] = np.mean(np.array(_store[j:,-2]).astype(float))
			#					   _store[:j, -2] = _store[:j, -2].astype(float) - ovrl_avg[i]['<']
			#					   _store[j:, -2] = _store[j:, -2].astype(float) - ovrl_avg[i]['>']
			#					   found = 1
			#					   print j
			#					   break
			#	   print "here"
			# 
			# if found != 1:
			#	   print "came here"
			#	   ovrl_avg[i]['<'] = np.mean(np.array(_store[:,-2]).astype(float))
			#	   _store[:, -2] = _store[:, -2].astype(float) - ovrl_avg[i]['<']
			# 
			# mean = np.mean(_store[:,-2].astype(float))
			# std = np.std(_store[:,-2].astype(float))
			# plt.plot(_store[:,-2].astype(float))
			# plt.suptitle("After OVRL"+" | MEAN: {0:.4f}".format(mean)+" | STD: {0:.4f}".format(std))
			# plt.show()
			
			## WEEK
			_split, _ = split_col(_store, 2)
			for x in _split:
				x = np.array(x)
				week_avg[i][x[0,2]] = np.mean(np.array(x[:,-2].astype(float)))
				for _x in x:
					_store[_x[-1],-2] = float(_store[_x[-1],-2]) - week_avg[i][x[0,2]]
			# mean = np.mean(_store[:,-2].astype(float))
			# std = np.std(_store[:,-2].astype(float))
			# print std
			# plt.plot(_store[:,-2].astype(float))
			# plt.suptitle(str(_store_num)+" | After WEEK"+" | MEAN: {0:.4f}".format(mean)+" | STD: {0:.4f}".format(std))
			# plt.show()
			
			## STHLDY
			_split, _ = split_col(_store, 3)
			for x in _split:
				x = np.array(x)
				sthldy_avg[i][x[0,3]] = np.mean(np.array(x[:,-2].astype(float)))
				for _x in x:
					_store[_x[-1],-2] = float(_store[_x[-1],-2]) - sthldy_avg[i][x[0,3]]
			# mean = np.mean(_store[:,-2].astype(float))
			# std = np.std(_store[:,-2].astype(float))
			# print std
			# plt.plot(_store[:,-2].astype(float))
			# plt.suptitle(str(_store_num)+" | After STHLDY"+" | MEAN: {0:.4f}".format(mean)+" | STD: {0:.4f}".format(std))
			# plt.show()
			
			## SCHLDY
			_split, _ = split_col(_store, 4)
			for x in _split:
				x = np.array(x)
				schldy_avg[i][x[0,4]] = np.mean(np.array(x[:,-2].astype(float)))
				for _x in x:
					_store[_x[-1],-2] = float(_store[_x[-1],-2]) - schldy_avg[i][x[0,4]]
					
			# mean = np.mean(_store[:,-2].astype(float))
			std = np.std(_store[:,-2].astype(float))
			# print std
			# plt.plot(_store[:,-2].astype(float))
			# plt.suptitle(str(_store_num)+" | After SCHLDY"+" | MEAN: {0:.4f}".format(mean)+" | STD: {0:.4f}".format(std))
			# plt.show()
			y = np.array(train_store[_i])[:,-1].astype(float)
			y_pred = np.array(_store[:,-2]).astype(float)
			y_pred = y - y_pred
			print "Score: "+str(evalScore(y, y_pred))
			_stds.append(std)
			_scores.append(evalScore(y, y_pred))
			
			# plt.scatter(_stds, _scores)
			
		# plt.show()
		del params[:]
		params.append(ovrl_avg)
		params.append(week_avg)
		params.append(sthldy_avg)
		params.append(schldy_avg)
	
	if transform:
		y_pred = [0]*len(X)
		ovrl_avg = params[0]
		week_avg = params[1]
		sthldy_avg = params[2]
		schldy_avg = params[3]
		for i in range(len(y_pred)):
			y_pred[i] += ovrl_avg[int(X[i,0])-1][X[i,1]]
			y_pred[i] += week_avg[int(X[i,0])-1][X[i,2]]
			y_pred[i] += sthldy_avg[int(X[i,0])-1][X[i,3]]
			y_pred[i] += schldy_avg[int(X[i,0])-1][X[i,4]]
			
		return np.array(y_pred)
		
	return None

def storeInfo(data):
	_data = zip(data[:,0],[str(x[1])+"-"+str(x[2])+"-"+str(x[3]) for x in data])
	_comp_date = [str(x[5])+"-"+str(x[4]) for x in store]
	_month_days = [('01',0),('02',31),('03',59),('04',90),('05',120),('06',151),('07',181),('08',212),('09',243),('10',273),('11',304),('12',334)]
	_ovrhd = [6,5,4]
	_intv_preset = [[4,3,2,1,4,3,2,1,4,3,2,1],[1,4,3,2,1,4,3,2,1,4,3,2],[2,1,4,3,2,1,4,3,2,1,4,3]]
	_promo2_date = []
	_presets = []
	for x in store:
		if x[8] == '':
			_promo2_date.append('')
			_presets.append(-1)
			continue
		if int(x[8]) < 2013:
			_promo2_date.append(str(x[8]))
		else:
			totdays = (int(x[7])-1)*7 + _ovrhd[int(x[8])-2013] + 1
			for y in _month_days[::-1]:
				if totdays > y[1]:
					mon = y[0]
					day = totdays - y[1]
					if day < 10:
						day = "0"+str(day)
					else:
						day = str(day)
					break
			_promo2_date.append(str(x[8])+"-"+mon+"-"+day)
		if x[9].startswith('J'):
			_presets.append(0)
		elif x[9].startswith('F'):
			_presets.append(1)
		else:
			_presets.append(2)
	_data_comp = [(x[1], _comp_date[int(x[0])-1]) for x in _data]
	_comp = [0 if x[0] > x[1] else 1 for x in _data_comp]
	_promo2 = []
	for x in _data:
		_store_id = int(x[0])
		if _presets[_store_id-1] == -1:
			_promo2.append(0)
			continue
		if x[1] > _promo2_date[_store_id-1]:
			_mon = int(x[1].split('-')[1])
			_promo2.append(_intv_preset[_presets[_store_id-1]][_mon-1])
		else:
			_promo2.append(0)
	_store_attrs = np.array([store[int(x[0])-1, [1,2]] for x in _data])
	_comp_dist = np.array([store[int(x[0])-1, 3] for x in _data])
	_v = np.c_[np.array(_store_attrs), np.array(_promo2)[np.newaxis].T]
	# _v = np.c_[_v, store[:,[4,5]]]
	# _v = np.c_[_v, store[:,[7,8,9]]]
	_v = np.c_[_v, np.array(_comp)[np.newaxis].T]
	_v = np.c_[_v, _comp_dist]
	
	### sales | cust info
	_mf_features = custInfo()
	_mff = np.array([_mf_features[int(x[0])-1, :] for x in _data])
	_v = np.c_[_v, _mff]
	
	### past sales info
	_past = pastInfo(np.c_[data[:,0], np.asarray(zip(*_data)[1]), data[:,4]])
	_v = np.c_[_v, _past]
	
	return _v

def custInfo():
	inp = open('matx.pkl', 'rb')
	r = pickle.load(inp)
	s = pickle.load(inp)
	c = pickle.load(inp)
	inp.close()
	
	trun = decomposition.TruncatedSVD(n_components=20, algorithm='arpack', n_iter=0, random_state=0, tol=0.0)
	_r = trun.fit_transform(r)
	_s = trun.fit_transform(s)
	_c = trun.fit_transform(c)
	
	return np.c_[_r, _s, _c]

def pastInfo(data):
	# Input format: <Store_id> <Date> <Sales>
	# Output format: <past n days sales>
	
	_n = 10
	
	_data = sorted(zip(np.arange(len(data)),data), key=lambda x:int(x[1][0]))
	_idx, _data = zip(*_data)
	_data = np.asarray(_data)
	_idx = np.asarray(_idx)
	
	prev = _data[0,0]
	counter = 0
	_res = np.zeros((_data.shape[0], _n))
	for i, x in enumerate(_data):
		if prev == x[0]:
			if counter < _n:
				counter += 1
			else:
				if float(x[2]) == 0:
					_data[i,2] = int(np.mean(_data[i-_n:i,2].astype(float)))
				_res[i] = _data[i-_n:i,2].astype(float)
				counter += 1
				continue
		else:
			counter = 1
		_avg = np.mean(_data[i-counter+1:i+_n,2].astype(float))
		_res[i,0:_n-counter+1] = int(_avg)
		_res[i,_n-counter+1:_n] = _data[i-counter+1:i,2].astype(float)
	
	_bins = [200*(i+1) for i in range(100)]
	for i in range(_res.shape[1]):
		_res[:,i] = np.digitize(_res[:,i], _bins)
	
	_dates = {x: i for i, x in enumerate(sorted(np.unique(_data[:,1])))}
	_dates_ = {i: x for i, x in enumerate(sorted(np.unique(_data[:,1])))}
	_sales = np.zeros((1115, len(_dates)))
	
	print len(_dates)
	assert len(_dates) == 990
	
	prev = int(_data[0,0])
	for i, x in enumerate(_data):
		if prev == int(x[0]):
			_sales[prev-1, _dates[x[1]]] = int(float(x[2]))
		else:
			prev = int(x[0])
	
	for i in range(1115):
		for j in range(len(_dates)):
			if _sales[i,j] == 0:
				if j+5 < len(_dates):
					_nxt = np.mean(_sales[i,j+1:j+6])
				if _nxt == 0:
					_sales[i,j] = int(random.uniform(0,1000)+_sales[i,j-365])
					continue
				if j-5 > -1:
					_ltr = np.mean(_sales[i,j-5:j])
				_sales[i,j] = int(np.mean([_nxt,_ltr]))
	
	_mod_sales = np.zeros((1115, len(_dates)+10))
	
	for i in range(1115):
		for j in range(_mod_sales.shape[1]):
			if j < 212:
				_mod_sales[i,j] = 0.8*_sales[i,j+365]+0.2*_sales[i,j+730]
			elif j < 365:
				_mod_sales[i,j] = _sales[i,j+365]
			elif j < 577:
				_mod_sales[i,j] = 0.8*_sales[i,j-365]+0.2*_sales[i,j+365]
			elif j < 730:
				_mod_sales[i,j] = _sales[i,j-365]
			else:
				_mod_sales[i,j] = 0.8*_sales[i,j-365]+0.2*_sales[i,j-730]
	
	_win_size = 10
	
	_res_y = np.zeros((_data.shape[0], _win_size*2))
	for i, x in enumerate(_data):
		_id = int(x[0]) - 1
		j = _dates[x[1]]
		if j-10 < 0:
			_res_y[i, :j] = np.mean(_sales[_id, :10])
			_res_y[i, j:20] = _sales[_id, :20-j]
		else: 
			_res_y[i] = _mod_sales[_id, j-10:j+10]
	
	return np.c_[_res, _res_y].astype(int)

_train_data = map(dateSplit, train[:,2])
_train = np.c_[np.array(_train_data), train[:,[1,5,6,7,8]]]			 # YEAR | MONTH | DAY | WEEKDAY | OPEN | PROMO | STHLDY | SCHLDY

_test_data = map(dateSplit, test[:,3])
_test = np.c_[np.array(_test_data), test[:,[2,4,5,6,7]]]

storeID_train = np.array(map(int,train[:,0]))
X_train = _train
y_train = train[:,[3,4]].astype(int)
print "storeID_train: "+str(len(storeID_train))
print "X_train: "+str(X_train.shape)
print "y_train: "+str(y_train.shape)

ID_test_sno = test[:,[0]]
ID_test = np.array(map(int,test[:,1]))
X_test = _test
print "ID_test_sno: "+str(len(ID_test_sno))
print "ID_test: "+str(len(ID_test))
print "X_test: "+str(X_test.shape)

### Training
cv_size = 34565
train_size = len(storeID_train) - cv_size
test_size = len(ID_test)

ID_CV = storeID_train[:cv_size]
X_CV = X_train[:cv_size, :]
y_CV = y_train[:cv_size, :]

ID_train = storeID_train[cv_size:]
X_train = X_train[cv_size:, :]
y_train = y_train[cv_size:, :]

_coarse = False
_coarse_fine = False
_fine = False

y_pred_train = np.zeros(train_size)
y_pred_CV = np.zeros(cv_size)
y_pred_test = np.zeros(test_size)

### Coarse ##############################################
# _store = storeInfo(np.r_[np.c_[ID_CV[np.newaxis].T,X_CV[:,[0,1,2]],y_CV],np.c_[ID_train[np.newaxis].T,X_train[:,[0,1,2]]],y_train],np.c_[ID_test[np.newaxis].T,X_test[:,[0,1,2]]np.zeros((ID_test.shape[0],2))]])
_store = storeInfo(np.r_[np.c_[ID_CV,X_CV[:,[0,1,2]],y_CV],np.c_[ID_train,X_train[:,[0,1,2]],y_train],np.c_[ID_test,X_test[:,[0,1,2]],np.zeros((ID_test.shape[0],2))]])
coarse_params = []
_y_pred_train = coarseModel(np.c_[ID_train, _store[cv_size:cv_size+train_size,3], X_train[:,[3,6,7]]], y_train[:,0], True, True, coarse_params)
_y_pred_CV = coarseModel(np.c_[ID_CV, _store[:cv_size,3], X_CV[:,[3,6,7]]], None, False, True, coarse_params)
_y_pred_test = coarseModel(np.c_[ID_test, _store[cv_size+train_size:,3], X_test[:,[3,6,7]]], None, False, True, coarse_params)
y_pred_train += _y_pred_train
y_pred_CV += _y_pred_CV
y_pred_test += _y_pred_test
print "Coarse Train Score: "+str(evalScore(y_train[:,0], y_pred_train))
print "Coarse CV Score: "+str(evalScore(y_CV[:,0], y_pred_CV))

## Saving Model
output = open('coarse_model.pkl', 'wb')
pickle.dump(coarse_params, output, -1)
pickle.dump(_y_pred_CV, output, -1)
pickle.dump(_y_pred_train, output, -1)
pickle.dump(_y_pred_test, output, -1)
output.close()

input = open('coarse_model.pkl', 'rb')
coarse_params = pickle.load(input)
_y_pred_CV = pickle.load(input)
_y_pred_train = pickle.load(input)
_y_pred_test = pickle.load(input)
input.close()
y_pred_CV = _y_pred_CV
y_pred_train = _y_pred_train
y_pred_test = _y_pred_test
cv_size = len(_y_pred_CV)
train_size = len(_y_pred_train)
test_size = len(_y_pred_test)
##########################################################

### Data One Hot Encoding ################################
X_CV = np.c_[X_CV, _store[:cv_size, :]]
X_train = np.c_[X_train, _store[:train_size, :]]
X_test = np.c_[X_test, _store[:test_size, :]]

sales = np.r_[X_CV, X_train, X_test]
ohe_params = []
sales = ohe(sales, True, False, ohe_params, [0,1,2,3,4,5,6,7,8,9,10,11])		# add .tocsr()

X_CV = sales[0:cv_size, :]
X_train = sales[cv_size:cv_size+train_size, :]
X_test = sales[cv_size+train_size:, :]

## Saving Data
output = open('features_nothot_past_yr.pkl', 'wb')
pickle.dump(ID_CV, output, -1)
pickle.dump(sparse.csr_matrix(X_CV.astype(float)), output, -1)
pickle.dump(y_CV, output, -1)
pickle.dump(ID_train, output, -1)
pickle.dump(sparse.csr_matrix(X_train.astype(float)), output, -1)
pickle.dump(y_train, output, -1)
pickle.dump(ID_test, output, -1)
pickle.dump(sparse.csr_matrix(X_test.astype(float)), output, -1)
pickle.dump(ID_test_sno, output, -1)
output.close()

input = open('features.pkl', 'rb')
ID_CV = pickle.load(input)
X_CV = pickle.load(input)
y_CV = pickle.load(input)
ID_train = pickle.load(input)
X_train = pickle.load(input)
y_train = pickle.load(input)
ID_test = pickle.load(input)
X_test = pickle.load(input)
ID_test_sno = pickle.load(input)
input.close()
###########################################################

### Split ##################################################
_split_train, _shuffle_idx_train = split_col(np.c_[ID_train, X_train.A, y_train], 0)
_split_CV, _shuffle_idx_CV = split_col(np.c_[ID_CV, X_CV.A, y_CV], 0)
############################################################

### CF - Single
_prev_train = 0
_prev_CV = 0

_y_pred_train = np.ones(train_size)
_y_pred_CV = np.ones(cv_size)

_cf_models = [None]*len(_split_train)
_alphas = [0]*len(_split_train)
for i, store_data in enumerate(_split_train):
	_store_id = int(store_data[0,0])
	print "Store ID: "+str(_store_id)
	_shuffle_idx_train_store = _shuffle_idx_train[_prev_train : _prev_train +len(store_data)]
	_shuffle_idx_CV_store = _shuffle_idx_CV[_prev_CV : _prev_CV +len(_split_CV[i])]
	_prev_train += len(_shuffle_idx_train_store)
	_prev_CV += len(_shuffle_idx_CV_store)
	
	ridge = linear_model.RidgeCV(alphas=[ 0.03, 0.1, 0.3, 1., 3.0, 10., 30], fit_intercept=True, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=True)
	ridge.fit(store_data[:,1:-2], y_train[_shuffle_idx_train_store,0] - y_pred_train[_shuffle_idx_train_store])
	
	_y_pred_train[_shuffle_idx_train_store] = ridge.predict(store_data[:,1:-2])
	_y_pred_CV[_shuffle_idx_CV_store] = ridge.predict(_split_CV[i][:,1:-2])
	
	_alphas[_store_id-1] = ridge.alpha_
	_cf_models[_store_id-1] = ridge

y_pred_train += _y_pred_train
y_pred_CV += _y_pred_CV
print "CF Train Score: "+str(evalScore(y_train[:,0], y_pred_train))
print "CF CV Score: "+str(evalScore(y_CV[:,0], y_pred_CV))

### CF - Whole
# ridge = linear_model.RidgeCV(alphas=[ 0.1, 1., 10. ], fit_intercept=True, 
# 		normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=True)
# ridge.fit(X_train, y_train[:,0] - y_pred_train)
# _y_pred_train = ridge.predict(X_train.A)
# _y_pred_CV = ridge.predict(X_CV.A)
# y_pred_train += _y_pred_train
# y_pred_CV += _y_pred_CV
# print "CF Train Score: "+str(evalScore(y_train[:,0], y_pred_train))
# print "CF CV Score: "+str(evalScore(y_CV[:,0], y_pred_CV))
	
### RF - Single
_prev_train = 0
_prev_CV = 0

_y_pred_train = np.ones(train_size)
_y_pred_CV = np.ones(cv_size)

_rf_models = [None]*len(_split_train)
_bst_itr = [0]*len(_split_train)

for i, store_data in enumerate(_split_train):
	
	_store_id = int(store_data[0,0])
	print "Store ID: "+str(_store_id)
	_shuffle_idx_train_store = _shuffle_idx_train[_prev_train : _prev_train +len(store_data)]
	_shuffle_idx_CV_store = _shuffle_idx_CV[_prev_CV : _prev_CV +len(_split_CV[i])]
	_prev_train += len(_shuffle_idx_train_store)
	_prev_CV += len(_shuffle_idx_CV_store)
	
	# mdl = ensemble.RandomForestRegressor(n_estimators=5, criterion='mse', max_depth=None, min_samples_split=2,
	# 	min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True,
	# 	oob_score=False, n_jobs=1, random_state=None, verbose=1, warm_start=False)
	
	# mdl = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=20,
	# 	subsample=1.0, min_samples_split=1, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
	# 	max_depth=3, init=None, random_state=2389, max_features=None, alpha=0.9, verbose=0,
	# 	max_leaf_nodes=None, warm_start=False)
	
	# mdl = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=10, 
	# 	silent=True, objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1,
	# 	max_delta_step=0, subsample=0.8, colsample_bytree=1, base_score=0.5, seed=0,
	# 	missing=None)
	
	# mdl.fit(store_data[:,1:-2], y_train[_shuffle_idx_train_store,0] - y_pred_train[_shuffle_idx_train_store])
	
	dtrain = xgb.DMatrix(np.array(store_data[:,1:-2]), label = y_train[_shuffle_idx_train_store,0] - y_pred_train[_shuffle_idx_train_store], feature_names = [str(i) for i in range(store_data[:,1:-2].shape[1])])
	dcv = xgb.DMatrix(np.array(_split_CV[i][:,1:-2]), label = y_CV[_shuffle_idx_CV_store, 0] - y_pred_CV[_shuffle_idx_CV_store], feature_names = [str(i) for i in range(_split_CV[i][:,1:-2].shape[1])])
	params = {'silent':1, 'objective':'reg:linear', 'subsample':0.5, 'bst:eta':0.1}
	params = {'silent':1, 'objective':'reg:linear', 'subsample':0.5, 'eta':0.1, 'colsample_bytree':1.0, 'max_depth':6}
	evallist = [(dtrain,'train'), (dcv,'cv')]
	
	# xgb.cv(params, dtrain, num_boost_round=100, nfold=3, metrics=(), obj=None, feval=None,
	# 	fpreproc=None, as_pandas=False, show_progress=None, show_stdv=True, seed=0)
	
	mdl = xgb.train(params, dtrain, num_boost_round=100, evals=evallist, obj=None,
		feval=None, early_stopping_rounds=10, evals_result=None, verbose_eval=False)
	
	_y_pred_train[_shuffle_idx_train_store] = mdl.predict(dtrain, ntree_limit=mdl.best_iteration)
	_y_pred_CV[_shuffle_idx_CV_store] = mdl.predict(dcv, ntree_limit=mdl.best_iteration)
	
	# _y_pred_train[_shuffle_idx_train_store] = mdl.predict(store_data[:,1:-2])
	# _y_pred_CV[_shuffle_idx_CV_store] = mdl.predict(_split_CV[i][:,1:-2])
	
	_rf_models[_store_id-1] = mdl
	_bst_itr[_store_id-1] = mdl.best_iteration

y_pred_train += _y_pred_train
y_pred_CV += _y_pred_CV
print "CF Train Score: "+str(evalScore(y_train[:,0], y_pred_train))
print "CF CV Score: "+str(evalScore(y_CV[:,0], y_pred_CV))

### XGB - Whole
dtrain = xgb.DMatrix(X_train, label = y_train[:,0] - (y_pred_train), feature_names = [str(i) for i in range(X_train.shape[1])])
dcv = xgb.DMatrix(X_CV, label = y_CV[:, 0] - (y_pred_CV), feature_names = [str(i) for i in range(X_CV.shape[1])])
params = {'silent':1, 'objective':'reg:linear', 'subsample':0.7, 'eta':0.3, 'colsample_bytree':0.7, 'max_depth':6}
evallist = [(dtrain,'train'), (dcv,'cv')]

# xgb.cv(params, dtrain, num_boost_round=100, nfold=3, metrics=(), obj=None, feval=None,
# 	fpreproc=None, as_pandas=False, show_progress=None, show_stdv=True, seed=0)

mdl = xgb.train(params,dtrain,num_boost_round=20,evals=evallist,obj=None,feval=None,early_stopping_rounds=None,evals_result=None,verbose_eval=True)
_y_pred_train = mdl.predict(dtrain)
_y_pred_CV = mdl.predict(dcv)

y_pred_train += _y_pred_train
y_pred_CV += _y_pred_CV
print "CF Train Score: "+str(evalScore(y_train[:,0], y_pred_train))
print "CF CV Score: "+str(evalScore(y_CV[:,0], y_pred_CV))
print "CF Train Score: "+str(evalScore(y_train[:,0], y_pred_train+_y_pred_train))
print "CF CV Score: "+str(evalScore(y_CV[:,0], y_pred_CV+_y_pred_CV))

### GB - Whole
# gb = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
# 		subsample=0.8, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# 		max_depth=6, init=None, random_state=2389, max_features=None, alpha=0.9, verbose=1,
# 		max_leaf_nodes=None, warm_start=False)
# gb.fit(X_train.A, y_train[:,0] - y_pred_train)
# _y_pred_train = gb.predict(X_train.A)
# _y_pred_CV = gb.predict(X_CV.A)
# y_pred_train += _y_pred_train
# y_pred_CV += _y_pred_CV
# print "Fine Train Score: "+str(evalScore(y_train[:,0], y_pred_train))
# print "Fine CV Score: "+str(evalScore(y_CV[:,0], y_pred_CV))

### Submission
_split_test, _shuffle_idx_test = split_col(np.c_[ID_test, X_test.A, np.zeros((test_size,2))], 0)
_prev_test = 0
_y_pred_test = np.zeros(test_size)
for i, store_data in enumerate(_split_test):
	_store_id = int(store_data[0,0])
	_shuffle_idx_test_store = _shuffle_idx_test[_prev_test : _prev_test +len(store_data)]
	_prev_test += len(_shuffle_idx_test_store)
	dtest = xgb.DMatrix(np.array(store_data[:,1:-2]), feature_names = [str(i) for i in range(store_data[:,1:-2].shape[1])])
	_y_pred_test[_shuffle_idx_test_store] = _rf_models[_store_id-1].predict(dtest)

y_pred_test += _y_pred_test

# dtest = xgb.DMatrix(X_test, feature_names = [str(i) for i in range(X_test.shape[1])])
# _y_pred_test = mdl.predict(dtest)
# y_pred_test += _y_pred_test

y_pred_test[y_pred_test < 0] = 0
with open('Submission.csv', 'wb') as output:
	writer = csv.writer(output)
	writer.writerow(['Id','Sales'])
	writer.writerows(np.c_[ID_test_sno, y_pred_test])
