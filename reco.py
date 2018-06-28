
# coding: utf-8

# In[2]:


from numpy import *


# In[3]:


#initialize number of users and videos from database
num_videos=20
num_users=4


# In[22]:


#create ratings matrix from database
ratings = random.randint(6, size = (num_videos, num_users))
ratings

#difficulty level varies from 1 to 5. 1 is too easy, 5 is too hard. 3 is
#the sweet spot'''


# In[23]:


#rated matrix tells whether a particular movie is rated or not
rated_matrix=(ratings!=0)
print rated_matrix


# In[24]:


#get current user index and current topic
#create an array of video indexes video_index[] of all videos belonging 
#to current topic
#such that ratings[user_index][video_index[i]]=0
#USER is new to the topic
curr_user_index=0
curr_topic=None
#ensure INTEGER ARRAY
curr_video_index=zeros((3),dtype=int)
curr_video_index[0],curr_video_index[1],curr_video_index[2]=[3,6,8]


# In[25]:


#get topic_cost of each video, with respect to curr_topic
#closer two topics are, lower the topic_cost
#create an appropriate function to calculate topic_cost with current_topic
#as argument

topic_cost=[0,1,1,0,2,0,0,2,0,0,1,1,0,1,2,0,1,0,1,1]
topic_cost=topic_cost+ones((size(topic_cost),))


# In[26]:


#get var for each user other than current_user
#var=error*topic_cost summed over each video
#var represents rmse*topic_cost. high var => less effect on prediction
# we can neglect those videos which current_user has not watched
def calc_var(curr_user,cost_array):
    var_array=zeros((num_users,))
    for user in range(var_array.size):
        if user==curr_user:
            continue
        total,sq_err,var,count=0.0,0.0,0.0,0
        for j in range (0,num_videos):
            if j in curr_video_index:
                continue
            if rated_matrix[j,user]==False:
                continue
            else:
                count=count+1
                sq_err+=((ratings[j,user]-ratings[j,curr_user])**2)
                var+=sq_err*cost_array[j]
        var_array[user]=var/count
    return var_array
            
            
            
        


# In[27]:


var_arr=calc_var(curr_user_index,topic_cost)
var_arr


# In[28]:


copy_matrix=ratings-3
copy_matrix


# In[29]:


def video_score(curr_user,var_array):
    prediction_matrix=zeros((size(curr_video_index),num_users))
    for i in range(size(curr_video_index)):
        for j in range(num_users):
            if j==curr_user:
                continue
            prediction_matrix[i,j]=copy_matrix[curr_video_index[i],j]
            prediction_matrix[i,j]/=(var_array[j]+1)
    return prediction_matrix
        
        


# In[30]:


pred_mat=video_score(curr_user_index,var_arr)
pred_mat


# In[31]:


video_prediction=zeros((pred_mat.shape[0]))
for i in range(size(video_prediction)):
    count,res=0,0.0
    for j in range(num_users):
        if j==curr_user_index:
            continue
        n=int(curr_video_index[i])
        if rated_matrix[n,j]==False:
            continue
        else:
            count+=1
            res+= pred_mat[i,j]
    if count!=0:
        video_prediction[i]=res/count
    else: 
        video_prediction[i]=-100
video_prediction

