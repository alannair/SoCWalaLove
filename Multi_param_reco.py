
# coding: utf-8

# In[78]:


from numpy import *


# In[79]:


#initialize number of videos and users
num_videos=20
num_users=5


# In[80]:


#initialize attributes
num_params=6
#[difficulty,relevance,complexity,length,production,engaging]


# In[81]:


#create ratings matrix from database
ratings = random.randint(6, size = (num_videos, num_users), dtype=int)
ratings


# In[82]:


#create params_matrix 
params_matrix = random.random(size=(num_videos,num_users,num_params))
params_matrix


# In[83]:


#convert params_matrix to unit vectors
for video in range(num_videos):
    for user in range(num_users):
        norm=linalg.norm(params_matrix[video,user])
        params_matrix[video][user][:] = [x / norm for x in params_matrix[video][user]]
params_matrix


# In[84]:


#load user params
user_params = random.random(size=(num_users,num_params))
user_params


# In[85]:


#convert params_matrix to unit vectors

for user in range(num_users):
    norm=linalg.norm(user_params[user])
    user_params[user][:] = [x / norm for x in user_params[user]]
user_params


# In[86]:


#rated matrix tells whether a particular movie is rated or not
rated_matrix=(ratings!=0)
rated_matrix


# In[87]:


#get current user index and current topic
#create an array of video indexes video_index[] of all videos belonging 
#to current topic
#such that ratings[user_index][video_index[i]]=0 implies nhy7
#USER is new to the topic
curr_user_index=0
curr_topic=None
video_index=[]
for i in range(num_videos):
    if rated_matrix[i][curr_user_index]==False:
        video_index.append(i)
video_index
    


# In[88]:


#get topic_cost of each video, with respect to curr_topic
#closer two topics are, lower the topic_cost
#create an appropriate function to calculate topic_cost with current_topic
#as argument

# '''topic_cost=random.randint(3,size=(num_videos,),dtype=int)
# topic_cost=topic_cost+ones((size(topic_cost),))
# topic_cost'''


# In[89]:


#get var for each user other than current_user
#var=error*topic_cost summed over each video
#var represents rmse*topic_cost. high var => less effect on prediction
# we can neglect those videos which current_user has not watched
def calc_user_var(curr_user,user_params):
    var_mat = zeros(num_users)
    for user in range(num_users):
        if user == curr_user:
            continue
        else:
            for x in range(num_params):
                var_mat[user] += (user_params[user][x] - user_params[curr_user][x])**2
            
    return var_mat
            


    
# In[90]:


user_var = calc_user_var(curr_user_index,user_params)
user_var


# In[91]:


#higher the var, lower the value of that user's prediction
user_val = [1 - x for x in user_var]
user_val


# In[92]:


video_rated = zeros(size(video_index))
p=0
for video in video_index:
    count=0
    for user in range(num_users):
        if user == curr_user_index:
            continue
        else:
            video_rated[p] += ratings[video][user]*user_val[user]
            if rated_matrix[video][user] == True:
                count+=1
    video_rated[p] /= count
    p+=1

video_rated

