import numpy as np
import json
import os

class preprocessing:
	def __init__(self,ratings_file='small_ratings.txt',links_file='small_links.txt', user_min_index=0, user_max_index=1000, movie_min_index=0, movie_max_index=1000):
		self.ratings_file=ratings_file
		self.links_file=links_file
		self.user_min_index=user_min_index
		self.user_max_index=user_max_index
		self.movie_min_index=movie_min_index
		self.movie_max_index=movie_max_index
		self.social_dict={}
		self.rating_dict={}
	
	def get_smaller_user(self,input_file, output_file):
		rf = open(input_file,"r") 
		wf = open(output_file,"w") 
		small_user_size =0
		for line in rf:
			src,dst = line.split()
			src, dst = int(src),int(dst)
			if (src<self.user_max_index and src>self.user_min_index) and (dst<self.user_max_index and dst>self.user_min_index):
				wf.write(line)
				small_user_size +=1
		rf.close()
		wf.close()
		print 'small user size', small_user_size

	def get_smaller_movie(self,input_file, output_file):
		rf = open(input_file,"r") 
		wf = open(output_file,"w") 
		small_movie_size=0
		for line in rf:
			arr = line.split()
			user,movie = int(arr[0]),int(arr[1])
			if (user<self.user_max_index and user>self.user_min_index) and (movie<self.movie_max_index  and movie>self.movie_min_index):
				wf.write(line)
				small_movie_size+=1
		rf.close()
		wf.close()
		print 'small movie size', small_movie_size

	def fill_dict(self):
		rating_f = open(self.ratings_file,"r") 
		link_f = open(self.links_file,"r") 
		
		for line in link_f:
			src,dst = line.split()
			src, dst = int(src),int(dst)
			try:
				self.social_dict[src].append(dst)
			except:
				self.social_dict[src]=[dst]


		for line in rating_f:
			user,movie,rating = line.split()
			user,movie,rating = int(user),int(movie), float(rating)
			
			try:
				self.rating_dict[user][movie]=rating
			except:
				self.rating_dict[user]={movie:rating}
		rating_f.close()
		link_f.close()

	def generate_train(self, train_csv='train.csv'):
		# get ready 
		self.fill_dict()
		# convert to dense format
		# write csv
		train_size=0
		nn_input_f = open(train_csv,'w')
		for query_user in xrange(self.user_min_index,self.user_max_index):
			if query_user in self.social_dict and query_user in self.rating_dict:
				social_list = self.social_dict[query_user]
				my_rating_list= self.rating_dict[query_user]

				social_movie_view = []
				for friend in social_list:
					if friend in self.rating_dict:
						# social_movie_view = social_movie_view + [x[0] for x in self.rating_dict[friend]]
						social_movie_view = social_movie_view + self.rating_dict[friend].keys()

				# we have sparse format
				# save sparse format
				if len(my_rating_list)>0 and len(social_movie_view)>0:
					my_sparse = my_rating_list
					social_sparse = {i:social_movie_view.count(i) for i in set(social_movie_view)}
					train_data = {'my':my_sparse, 'social':social_sparse}
					json.dump(train_data, nn_input_f)
					nn_input_f.write(os.linesep)
					train_size+=1

				# convert to dense input of my rating vector
				# if len(my_rating_list)>0 and len(social_movie_view)>0:
				# 	# print my_rating_list
				# 	my_vector_dense = np.zeros(movie_len)
				# 	for m,r in my_rating_list:
				# 		my_vector_dense[m]=r
					
				# 	# convert to dense input of social movie view vector
				# 	social_vector_dense = np.zeros(movie_len);
				# 	for i in social_movie_view:
				# 		social_vector_dense[i]+=1


				# 	input_output_concat = np.concatenate((my_vector_dense,social_vector_dense, (my_vector_dense>0).astype(float)), axis=0)
				# 	np.savetxt(nn_input_f,input_output_concat.reshape(1,len(input_output_concat)), delimiter=',')
				# 	train_size+=1

		nn_input_f.close()
		print train_csv, 'size', train_size



# the number of node in links.txt <= 1049491 (1M)??
# the number of edge in links.txt = 7058819 (7M)
# the number of ratings in ratings.txt = 8196077 (8M)

user_len = 100*1000
movie_len = 10*1000
train_file = 'train_'+str(user_len/1000)+'_'+str(movie_len/1000)+'.csv'
small_ratings_file = 'small_ratings'+str(user_len/1000)+'_'+str(movie_len/1000)+'.txt'
small_links_file = 'small_links_'+str(user_len/1000)+'_'+str(movie_len/1000)+'.txt'

		
pp = preprocessing(ratings_file=small_ratings_file,links_file=small_links_file, user_min_index=0, user_max_index=user_len, movie_min_index=0, movie_max_index=movie_len)

pp.get_smaller_movie('ratings.txt', small_ratings_file)

pp.get_smaller_user('links.txt', small_links_file)


pp.generate_train(train_file)