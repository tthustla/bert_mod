# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# set the names for each column
cols = ['sentiment','id','date','query_string','user','text']
def main():
	# read training data with ISO-8859-1 encoding and column names set above
	df = pd.read_csv('temp/training.1600000.processed.noemoticon.csv', encoding = 'ISO-8859-1',names=cols)
	msk = pd.read_csv('filter.csv')['index_to_filter']
	df = df.loc[df.index.difference(msk),['sentiment','text']]
	# shuffle the data
	label_map = {4: 1}
	df = df.replace({"sentiment": label_map})
	df['text'] = [x.encode('utf8') for x in df.text]
	# set the random seed and split train and test with 99 to 1 ratio
	SEED = 2000
	train, dev_and_test = train_test_split(df, test_size=.02, random_state=SEED)
	dev, test = train_test_split(dev_and_test, test_size=.5, random_state=SEED)
	# np.random.seed(777)
	# msk = np.random.rand(len(df)) < 0.99
	# train = df[msk].reset_index(drop=True)
	# dev = df[~msk].reset_index(drop=True)
	# save both train and test as CSV files
	train.to_csv('train.tsv',sep='\t', header=False)
	dev.to_csv('dev.tsv',sep='\t', header=False)
	test.to_csv('test.tsv',sep='\t', header=False)

if __name__=="__main__":
	main()