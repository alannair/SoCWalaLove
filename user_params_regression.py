
# coding: utf-8

# In[15]:


from numpy import *
from pandas import *
import matplotlib.pyplot as plt


# In[16]:


#predict optimal params_list for curr_user

#load content history for current history
#initialize number of videos in database
num_videos=20

curr_user_ratings = random.randint(6, size = (num_videos,))

num_params=6

curr_user_ratings
#[difficulty,relevance,complexity,length,production,engaging]


# In[17]:


#load video params list

video_params = random.random(size=(num_videos,num_params))
video_params


# In[18]:


#convert params_matrix to unit vectors, skip if db is already of this form
for video in range(num_videos):
    norm=linalg.norm(video_params[video])
    video_params[video][:] = [x / norm for x in video_params[video]]
video_params


# In[21]:


sample = zeros(num_params)

def linear_regression(x, y, m_current=sample, b_current=0, epochs=100000, learning_rate=0.001, approx0=0.001):
    N = float(len(y))
    for iteration in range(epochs):
        
        
        y_pred = zeros(num_videos)
        m_grad = sample
        b_grad = 0
        for i in range(num_videos):
            for j in range(num_params):
                y_pred[i] +=  m_current[j]*x[i,j]
            y_pred[i] += b_current
                
        cost = sum([data**2 for data in (y-y_pred)]) / N
        
       
        for var_video in range(num_videos):
            for var_param in range(num_params):
                m_grad[var_param] = (-2/N)* (x[var_video,var_param]*(y[var_video]-y_pred[var_video]))
            b_grad = (-2/N)*(y[var_video]-y_pred[var_video])
            
        m_current[:] -= [(learning_rate * dead) for dead in m_grad]
        b_current -= (learning_rate * b_grad)
#         print cost
#         print m_current
#         norm=linalg.norm(m_grad)
#         if norm > approx0 and norm < 10*approx0:
#             return m_current, b_current, cost
# #        print cost
        
            
        
#         y_current = (m_current * X) + b_current
#         cost = sum([data**2 for data in (y-y_current)]) / N
#         m_gradient = -(2/N) * sum(X * (y - y_current))
#         b_gradient = -(2/N) * sum(y - y_current)
#         m_current = m_current - (learning_rate * m_gradient)
#         b_current = b_current - (learning_rate * b_gradient)
    return m_current, b_current, cost


# In[22]:


linear_regression(video_params,curr_user_ratings)


# In[51]:


plt.plot(video_params[:,1],curr_user_ratings,'*')

