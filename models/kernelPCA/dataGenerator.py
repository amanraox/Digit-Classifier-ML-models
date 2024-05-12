# Code to generate data after applying kPCA
from helpers import *

train_dataset = dsets.MNIST(root='../../utils/data', train=True, download=True, transform=transforms.ToTensor())
train_dataset = train_dataset.data.reshape(-1,784)
len1 = len(train_dataset)

data1 = train_dataset[0:7500,:]
data2 = train_dataset[7500:15000,:]
data3 = train_dataset[15000:22500,:]
data4 = train_dataset[22500:30000,:]
data5 = train_dataset[30000:37500,:]
data6 = train_dataset[37500:45000,:]
data7 = train_dataset[45000:52500,:]
data8 = train_dataset[52500:60000,:]

print('Starting kpca', flush = True)
kpca = KernelPCA(kernel = "rbf", gamma = 0.000001 ,n_components = 200)


print('Transforming data1', flush=True)
data1 = kpca.fit_transform(data1)
print('Transforming data2', flush=True)
data2 = kpca.fit_transform(data2)
print('Transforming data3', flush=True)
data3 = kpca.fit_transform(data3)
print('Transforming data4', flush=True)
data4 = kpca.fit_transform(data4)
print('Transforming data5', flush=True)
data5 = kpca.fit_transform(data5)
print('Transforming data6', flush=True)
data6 = kpca.fit_transform(data6)
print('Transforming data7', flush=True)
data7 = kpca.fit_transform(data7)
print('Transforming data8', flush=True)
data8 = kpca.fit_transform(data8)

print('Finished kpca', flush = True)

full_data = np.vstack((data1, data2, data3, data4, data5, data6, data7, data8))

validation_dataset = dsets.MNIST(root='../../utils/data', train=False, download=True, transform=transforms.ToTensor())
validation_dataset = validation_dataset.data.reshape(-1,784)

data1 = validation_dataset[0:5000, :]
data2 = validation_dataset[5000:10000, :]

print('\nNow for the validation data\n\nTransforming data1', flush=True)
data1 = kpca.transform(data1)
print('Transforming data2', flush=True)
data2 = kpca.transform(data2)

val_data = np.vstack((data1, data2))

store = {'data':full_data, 'kpca':kpca, 'validation':val_data}

with open('model/kernel_pca_polynomial.pkl', 'wb') as handle:
	pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)

