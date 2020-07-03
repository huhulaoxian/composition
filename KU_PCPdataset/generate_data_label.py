#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

class GenDataLabel():
    	
	def checkData(self,path,num):
		filelist =	(path)
		for i in range(num):
			name = str(i+1).zfill(4) + '.jpg'
			if os.path.isfile(filelist+"/"+name)==False:
				print (name,os.path.isfile(filelist+"/"+name))
				
	def genSinclassList(self,path_list,listname,label_list,class_num):
		with open(label_list, 'r') as l:
			with open(listname,'w') as f:
				i = 0
				while i < len(path_list):
					label = l.readline().replace(' ','')
					print(label)
					f.write(os.path.join(path,path_list[i])+' '+label[class_num]+'\n')
					i = i+1
	
	def genMultilabelList(self,path_list,listname,label_list):
		with open(label_list, 'r') as l:
			with open(listname,'w') as f:
				i = 0
				while i < len(path_list):
					label = l.readline().replace(' ','')
					f.write(os.path.join(path,path_list[i])+' '+label)
					i = i+1

	def genSinlabelList(self,path_list,listname,label_list):
		with open(label_list, 'r') as l:
			with open(listname,'w') as f:
				i = 0
				label = 1
				while label:
					label = l.readline().replace(' ','')
					if label.count('1') ==0:
						i = i+1
						continue
					print('label',i,label)
					f.write(os.path.join(path,path_list[i])+' '+ str(label.index('1'))+'\n')
					i = i+1

	def stasistic(self,path_list,filename,label_list):
		with open(label_list, 'r') as l:
			with open(listname,'w') as f:
				i = 0
				one_label = 0
				two_label = 0
				tree_label = 0
				other_label = 0
				
				while i < len(path_list):
					label = l.readline().replace(' ','')
					if label.count('1')  == 1:
						one_label = one_label+1
					if label.count('1')  == 2:
						two_label = two_label+1
					if label.count('1')  == 3:
						tree_label = tree_label+1
					if label.count('1')  == 0:
						other_label = other_label+1
					i = i+1
				f.write('one_label:'+str(one_label)+'\n'+'two_label:'+str(two_label)+'\n'
				+'tree_label:'+str(tree_label)+'\n'+'other_label:'+str(other_label)+'\n')
				
if __name__ == '__main__':
	#train or test
	train_test = 'train'
	
	path = './'+train_test+'_img/'
	label_list = './'+train_test+'_label.txt'
	path_list= os.listdir(path)
	path_list.sort() 
	num = 1085
	genDataLabel = GenDataLabel()
	#check the number of data
	#genDataLabel.checkData(path,num)
	
	#generate the filelist of single classifier
	class_nums = [0,1,2,3,4,5,6,7,8]	#0-8
	for class_num in class_nums:
		listname = train_test+'_list_'+ str(class_num) +'.txt'
		genDataLabel.genSinclassList(path_list,listname,label_list,class_num)
	
	#generate the filelist of multiple_lable
	listname = train_test+'_list_multi.txt'
	genDataLabel.genMultilabelList(path_list,listname,label_list)
	
	#generate the filelist of single_lable
	listname = train_test+'_list_single.txt'
	genDataLabel.genSinlabelList(path_list,listname,label_list)
	
	#stasistic the number of label of image
	listname = train_test+'stasistic.txt'
	genDataLabel.stasistic(path_list,listname,label_list)
