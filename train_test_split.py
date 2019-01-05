# import libraries
import pandas as pd
import numpy as np

# set the names for each column
cols = ['sentiment','id','date','query_string','user','text']
def main():
	# read training data with ISO-8859-1 encoding and column names set above
	df = pd.read_csv('temp/training.1600000.processed.noemoticon.csv', encoding = 'ISO-8859-1',names=cols)
	# shuffle the data
	df = df.sample(frac=0.01).reset_index(drop=True)
	label_map = {4: 1}
	df = df.replace({"sentiment": label_map})
	# set the random seed and split train and test with 99 to 1 ratio
	np.random.seed(777)
	msk = np.random.rand(len(df)) < 0.9
	train = df[msk].reset_index(drop=True)
	dev = df[~msk].reset_index(drop=True)
	# save both train and test as CSV files
	train[['sentiment','text']].to_csv('train.tsv',sep='\t', header=False)
	dev[['sentiment','text']].to_csv('dev.tsv',sep='\t', header=False)

if __name__=="__main__":
	main()