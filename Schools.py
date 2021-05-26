#!/usr/bin/env python
# coding: utf-8

# # Read in the data

# In[1]:


import pandas as pd
import numpy
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]

data = {}

for f in data_files:
    d = pd.read_csv("schools/{0}".format(f))
    data[f.replace(".csv", "")] = d


# # Read in the surveys

# In[2]:


all_survey = pd.read_csv("schools/survey_all.txt", delimiter="\t", encoding='windows-1252')
d75_survey = pd.read_csv("schools/survey_d75.txt", delimiter="\t", encoding='windows-1252')
survey = pd.concat([all_survey, d75_survey], axis=0)

survey["DBN"] = survey["dbn"]

survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_11", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]
survey = survey.loc[:,survey_fields]
data["survey"] = survey


# # Add DBN columns

# In[3]:


data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]

def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return "0" + string_representation
    
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(pad_csd)
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]


# # Convert columns to numeric

# In[4]:


cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

def find_lat(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat

def find_lon(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon

data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"], errors="coerce")
data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"], errors="coerce")


# # Condense datasets

# In[5]:


class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]

class_size = class_size.groupby("DBN").agg(numpy.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size

data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]


# # Convert AP scores to numeric

# In[6]:


cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    data["ap_2010"][col] = pd.to_numeric(data["ap_2010"][col], errors="coerce")


# # Combine the datasets

# In[7]:


combined = data["sat_results"]

combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

to_merge = ["class_size", "demographics", "survey", "hs_directory"]

for m in to_merge:
    combined = combined.merge(data[m], on="DBN", how="inner")

combined = combined.fillna(combined.mean())
combined = combined.fillna(0)


# # Add a school district column for mapping

# In[8]:


def get_first_two_chars(dbn):
    return dbn[0:2]

combined["school_dist"] = combined["DBN"].apply(get_first_two_chars)


# # Find correlations

# In[9]:


correlations = combined.corr()
correlations = correlations["sat_score"]
print(correlations)


# # Plotting survey correlations

# In[10]:


# Remove DBN since it's a unique identifier, not a useful numerical value for correlation.
survey_fields.remove("DBN")


# In[11]:


correlations[survey_fields].plot.bar()


# The `N_s` and `N_p` are the first columns to highlight our view. But it makes sense to have high correlation with the `sat_score` column since these columns talk about of the enrollment the people have in the schools.
# 
# `aca_s` is another one with high correlation, this talks about the academic expectation scores from the students. The expectations affects the students, but it does not apply to the teachers or parents, which are the `aca_t_11` and  `aca_p_11` 
# 
# We can see that te `saf_t_11` and `saf_s_11` have a high correlation with the `sat_score`. These columns talks about the score the teahcers and students put in the safety and respect inside the school. It is hard to learn or teach in violent enviroments, so this correlation may be normal.
# 
# Based on this, we will dig and figure out which schools have low safety scores.

# # Analyzing safety
# 
# Lets plot the correlation between the SAt Score and the safety. Each dot is a school.

# In[12]:


combined.plot.scatter(x='saf_s_11',y='sat_score')


# Watching the plot, we can see that the safety affects in some manner the Sat Score, but not affects very strong. For a lot of `Safety scores`, there are `SAT Scores` rounding 1000-1200, even with a Safety score of `8.0`.
# 
# But it is clear that low `Safety Scores` rarely achieve high `SAT Scores`. We have the range of `Safety` from 5 to 9, the `SAT Score` boundary for low `Safety` Schools in range `5 to 6.7` is around `1500`, it is not until a 7.0 of `Safety` that we start watching regularly higher `SAT Scores`.

# ## Borough Safety

# In[13]:


borough=combined.groupby('boro')
borough=borough['saf_s_11'].agg(numpy.mean).sort_values()
borough


# Looks like `Brooklyn` and `Staten Iland` are the Boroughs where safety is the lowest, while Manhattan and Queens are the top.

# ## Race SAT Performance
# 
# In the dataset we have columns that tell us the percetage of each race at every school:
# 
# - white_per
# - asian_per
# - black_per
# - hispanic_per
# 
# Plotting the correlation of these columns with the `sat_score`, we can determine if there are any racial difference in SAT performance.

# In[14]:


races=['white_per','asian_per','black_per','hispanic_per']
race= combined.corr()
race=race.loc[races,'sat_score']
print(race)
race.plot.bar()


# Srprisingly there are strong correlations between the Sat Score for 2 races:
# - White
# - Asian
# 
# If there are mor percentage of one race in the school it may increas the SAT performance.
# 
# The correlations in black and hispanic columns are negative, this may mean that when the percentage of some of these races go up, the SAT score decrease and viceversa.
# 
# To have a better understanding lets explore schools with low SAT scores and high values for `hispanic_per`

# In[15]:


print(combined.plot.scatter(x='hispanic_per',y='sat_score'))


# We can see a negative correlation between the 2 variables, the correlation is not very strong, but we can see the slightly the pattern.
# 
# Indeed, those schools that have  low percentage of hispanic people are the ones who has the SAT Score higher, while the percentage of hispanic people increase, the SAT Score decrease, until the point that startin from 25% of hispanic people in a school, the SAT Score doesn's surpass 1600 points.

# In[16]:


hispa=combined[combined['hispanic_per']>95]
hispa=hispa['SCHOOL NAME']
hispa


# The schools above are schools with 95% or more of hispanic people, these schools are schools where immigrant people come to learn, this means most of the students are learning English, this would explain the low SAT Scores.

# In[17]:


hisp_less=combined[(combined['hispanic_per']<10)&(combined['sat_score']>1800)]['SCHOOL NAME']
hisp_less


# After some research, it was found thar that these schools are science and tehcnical schools, which require a previous test to be able to join. This doesn't explain the low ratio of hispanic people, but explains why they have a high SAT Score.

# ## Gender difference in SAT Score

# In[21]:


gender=combined.corr()
gender=gender.loc[['male_per','female_per'],'sat_score']
print(gender)
print(gender.plot.bar())


# Both gender has almost the same correlation, the correlation is not strong being:
# 
# - Men: -0.112
# - Women: 0.012
# 
# This implies that a high percentage of men in a school correlates negatively with the SAT Score, while a high percentage of women at school implies a positive correlation with the SAT Score.

# In[22]:


fe_corr=combined.plot.scatter(x='female_per',y='sat_score')


# Looking at the scatter plot, does not seem to be any correlation between the columns, but there is a range of percentage girls at school between 60% and 80% with hoigh SAT Score.
# 
# The name of these schooles are:

# In[24]:


fe_great=combined[(combined['female_per']>60)&(combined['sat_score']>1700)]['SCHOOL NAME']
fe_great


# These schools only accept students or by making a test and selecting the best ones or by a rough process of little tasks, in resume, high standars. Also they are liberal art schools.

# ## AP Scores vs SAT Scores
# 
# AP is a test that students can take for some topic at the end of school. Lets see how many people took these tests and then make the correlation with the SAT Score.

# In[30]:


combined['ap_per']=combined['AP Test Takers ']/combined['total_enrollment']
combined['ap_per']
combined.plot.scatter(x='ap_per',y='sat_score')


# It doesn't look like there is a correlation, or not a strong one.  But we can see how while the percentage of sutendt taking the AP test increase, there are some schools that have high SAT Scores, but this dissapear when the `ap_per` reaches the 60%.

# In[ ]:




