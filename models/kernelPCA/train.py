from helpers import *

torch.manual_seed(15)

parser = argparse.ArgumentParser(description='Image Detection')
parser.add_argument('-use_trained_model', action = 'store_true')

dataset = Data()

# Define analysis function
def analyse():
	correct = 0
	incorrect = 0
	for i in range(len(validation_dataset)):
		x, y = dataset.get_val(i)
		z = model(torch.from_numpy(x.reshape(-1, img_dim)).float())
		_, yhat = torch.max(z, 1)
		if(yhat == y):
			correct += 1
		else:
			incorrect += 1

	return (correct, incorrect, correct/(correct+incorrect))


args = parser.parse_args()

if(not args.use_trained_model):

	# Define a criterion function
	criterion = nn.CrossEntropyLoss()

	# Create Dataloader objects
	trainloader = DataLoader(dataset = dataset, batch_size = 500, shuffle=True)

	# Define model parameters and create a model
	# The Neural Network will have a single hidden layer with 100 neurons
	in_dim = img_dim
	out_dim = 10
	Hidden1 = 1000
	Hidden2 = 150
	# Hidden3 = 200
	# Hidden4 = 200
	model = Neural_Network(in_dim, Hidden1, Hidden2, out_dim)

	# Define an optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

	# train the model now!! (on 10 epochs)
	epochs = 100

	prev_acc = 0
	for epoch in range(epochs):
		if((epoch) % 5 == 0 or prev_acc > .95):
			(C, I, A) = analyse()
			prev_acc = A
			print("Accuracy =", A, flush=True)
		print('Running on epoch {}'.format(epoch + 1), flush = True)
		for (x,y) in trainloader:
			optimizer.zero_grad()
			z = model(x.view(-1,img_dim).float())
			loss = criterion(z, y)
			loss.backward()
			optimizer.step()


	with open('model/trained_model.pkl', 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
	if(not os.path.isfile('model/trained_model.pkl')):
		print('Train the model first')
		os._exit(1)

	with open('model/trained_model.pkl', 'rb') as f:
		model = pickle.load(f)	

# Count the classified and miss classified data using the validation set
(C, I, A) = analyse()

print("Analysis:")
print("Correctly classified data count =", C)
print("Incorrectly classified data count =", I)
print("Accuracy =", A)

plt.show()