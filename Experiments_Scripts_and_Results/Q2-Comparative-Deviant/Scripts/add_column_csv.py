#from util.csvProcessing import writeCSV,writeCSVwithHeader,readCSVwithHeader
import csv
from gc import collect
import argparse

def readCSVwithHeader(source,selectedHeader=None,numberHeader=None,arrayHeader=None,delimiter='\t'):
	results=[]
	header=[]
	count=0
	
	with open(source, 'rb') as csvfile:
		
		readfile = csv.reader(csvfile, delimiter=delimiter)
		
		header=next(readfile)
		
		
		#selectedHeader=selectedHeader if selectedHeader is not None else header
		
		range_header=range(len(header))
		
		if numberHeader is None and arrayHeader is None  :
			if selectedHeader is None:
				results=[{header[i]:row[i] for i in range_header} for row in readfile]
			else :
				results=[{header[i]:row[i] for i in range_header if header[i] in selectedHeader} for row in readfile]
		else :
			numberHeader=numberHeader if numberHeader is not None else []
			arrayHeader=arrayHeader if arrayHeader is not None else []
			if selectedHeader is None:
				results_append=results.append
				for row in readfile:
					elem={}
					skip=False
					for i in range_header:
						if header[i] in numberHeader:
							try :
								elem[header[i]]=float(row[i])
							except:
								skip=True
						elif header[i] in arrayHeader:
							elem[header[i]]=eval(row[i])
						else :
							elem[header[i]]=row[i]
					if not skip:
						results_append(elem)
					
				#results=[{header[i]:float(row[i]) if header[i] in numberHeader else eval(row[i]) if header[i] in arrayHeader else row[i] for i in range_header} for row in readfile ] 
			else :
				results=[{header[i]:float(row[i]) if header[i] in numberHeader else eval(row[i]) if header[i] in arrayHeader else row[i] for i in range_header if header[i] in selectedHeader} for row in readfile]

	collect()
	return results,header

def writeCSVwithHeader(data,destination,selectedHeader=None,delimiter='\t',flagWriteHeader=True):
	header=selectedHeader if selectedHeader is not None else data[0].keys()
	
	if flagWriteHeader : 
		with open(destination, 'w') as f:
			f.close()
	with open(destination, 'ab+') as f:
		writer2 = csv.writer(f,delimiter='\t')
		if flagWriteHeader:
			writer2.writerow(header)
		for elem in iter(data):
			row=[]
			for i in range(len(header)):
				row.append(elem[header[i]])
			writer2.writerow(row)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='DEPICT-XPs')
	parser.add_argument('file', metavar='ConfigurationFile', type=str,  help='the input configuration file')
	args=parser.parse_args()

	csv_file_path=args.file
	d,h=readCSVwithHeader(csv_file_path)
	for row in d:
		prune=eval(row['pruning'])
		using_approx=eval(row['using_approxmations'])
		bootstraping_ci=eval(row['compute_bootstrap_ci'])
		print prune,using_approx,bootstraping_ci
		algo='NAIVE'
		if using_approx:
			if bootstraping_ci:
				if prune:
					algo='DEPICT_BS'
				else:
					algo='DEPICT_BS_NO_PRUNE'
			else:
				if prune:
					algo='DEPICT'
				else:
					algo='DEPICT_NO_PRUNE'
		else:
			if bootstraping_ci:
				if prune:
					algo='NAIVE_BS_PRUNE'
				else:
					algo='NAIVE_BS'
			else:
				if prune:
					algo='NAIVE_PRUNE'
				else:
					algo='NAIVE'
		#raw_input('......')
		#print row
		row['algorithm']=algo
		row['timespent']=row['timespent_total']

	writeCSVwithHeader(d,csv_file_path,["algorithm"]+h+["timespent"])
		
	