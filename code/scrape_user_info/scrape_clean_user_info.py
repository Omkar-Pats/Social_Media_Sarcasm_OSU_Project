'''
Code for scraping user info from reddit
'''

##########################################################################################
###################################### Imports ###########################################

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent
from random import randint
from tqdm import tqdm
import dateutil.parser as dp
import multiprocessing as mp
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup



##########################################################################################
############################# Load and Basic Data Prep ###################################

df = pd.read_csv('data/base_data/base-data-sarcasm.csv')

users = df['author'].drop_duplicates()
print(users.head(), end="\n\n")



#####################################################################################################
############################# Initializations and Helper Functions ##################################

# setting a dry run variable becasue this code takes around 20 mins to run completely
# setting this to True will run only for a few users and will complete within seconds
# existing user file will not be overwritten. Keep Default as True
DRY_RUN = True

USER_PAGE_BASE_URL = 'https://old.reddit.com/user/'

ua = UserAgent()
UA_LIST = [ua.chrome, ua.firefox, ua.safari, ua.opera, ua.edge]

ERRORS_LIST = []

ERROR_LOG_STRING_FORMAT = '''
    {section_title}
    ---
    {description}
    ---
    {error_message}
''' + '='*60 + '\n\n'

### helper function for logging errors while scraping or cleaning
def error_logging_helper(log_filename:str = 'all_logs.log', logger_name:str = 'error_logger'):
    '''
    Helper function for logging errors
    Inputs:
    - log_filename: string, filename where to log the errors
    - logger_name: string, name of the logger initialized
    '''
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(
        filename=log_filename,
        mode='a+'
    )
    log_format = logging.Formatter('%(asctime)s - %(message)s')
    handler.setLevel(logging.INFO)
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    print(logger.getEffectiveLevel())

    return logger


LOG_FILENAME = 'assets/all_logs.log'
logger = error_logging_helper(LOG_FILENAME)


### helper function for asynchronous scraping
def async_request_extract_html_wrapper(url:str):
    '''
    wrapper for scraping and extracting html element in an asynchronous manner
    increases speed

    Inputs:
        - url: string containing the full url
    '''
    try:
        headers = {
            'User-Agent':UA_LIST[randint(0,4)]
        }

        rq_result = rq.get(
            url,
            headers = headers,
            timeout = 10
        )

        return rq_result
    
    except Exception as e:
        err = (url, str(e))
        ERRORS_LIST.append(err)


### helper function for using multiprocessing to cleaning scraped html data
def extract_and_clean_info_from_html_mp(rq_result):
    '''
    helper function for extracting user info from the scraped html content
    Inputs:
        rq_result: request object obtained from requests.get
    '''
    try:
        html_result = BeautifulSoup(rq_result.content, "html.parser")

        ### extract post karma obtained by user
        extracted_elem = html_result.find(
            'span', {'class':'karma'}
        )
        if extracted_elem is not None:
            post_karma = extracted_elem.text.replace(',', '')
            post_karma = int(post_karma)
        else:
            post_karma = 0


        ### extract comments karma obtained by user
        extracted_elem = html_result.find(
            'span', {'class':'comment-karma'}
        )
        if extracted_elem is not None:
            comment_karma = extracted_elem.text.replace(',', '')
            comment_karma = int(comment_karma)
        else:
            comment_karma = 0


        ### extract user joining date
        join_date = html_result.find(
            'time'
        )
        if join_date is not None:
            join_date = dp.parse(join_date['datetime']).date()
        else:
            join_date = dp.parse('17 March 2023').date()


        ### extract number of comments/posts the user received awards for - gilded posts and comments
        all_gilded_posts = html_result.find_all(
            'div', attrs={'class':['gilded', 'thing']}
        )
        num_gilded_posts = len(all_gilded_posts)


        ### extract the list of subreddits the user received awards in
        gilded_subreddits_list = [i['data-subreddit'] for i in all_gilded_posts]
   
    except Exception as err_str:
        
        ### extract the list of subreddits the user received awards in
        logger.error(
            ERROR_LOG_STRING_FORMAT.format(
                section_title = 'Error in Extraction and Cleaning',
                description = '',
                error_message = str(err_str) + "\n" + str(traceback.format_exc())
            )
        )
        return [0, 0, dp.parse('17 March 2023').date(), 0, []]
    
    return [post_karma, comment_karma, join_date, num_gilded_posts, gilded_subreddits_list]



if __name__ == '__main__':
    #####################################################################################################
    ############################# Scraping - Uses Asynchronous Process ##################################

    url_list = [USER_PAGE_BASE_URL + username + '/gilded' for username in users]
    print("Total No. of URLs: ", len(url_list), end="\n\n")

    ### converting into chunks - so that we don't lose progress in case of error
    n = 1000
    url_list_chunks = [url_list[i:i + n] for i in range(0, len(url_list), n)]

    print("Size Single Chunk: ", len(url_list_chunks[0]))
    print("Total Chunks: ", len(url_list_chunks), end="\n\n")


    ### asynchronous request calls - fetch data for multiple users parallely for faster processing
    start = time.time()

    if DRY_RUN:
        dry_run_index1 = 2
    else:
        dry_run_index1 = len(url_list_chunks)

    all_results_list = []

    for single_batch_url_list in tqdm(url_list_chunks[0:dry_run_index1]):
        event_loop = asyncio.get_event_loop()
        
        # asynchronous call to the wrapper function
        Executor = ThreadPoolExecutor(max_workers=len(single_batch_url_list))
        tasks = [event_loop.run_in_executor(Executor, async_request_extract_html_wrapper, url) for url in single_batch_url_list]

        #single_batch_result = await asyncio.gather(*tasks)
        single_batch_result = event_loop.run_until_complete(asyncio.gather(*tasks))

        all_results_list.extend(single_batch_result)
        
    print("Total time taken for scraping: ", time.time() - start, end="\n\n")

    print("Scraped Results Summary: ")
    print(len(all_results_list))
    print(sum([i.status_code == 200 for i in all_results_list if i is not None]))
    print(sum([i is None for i in all_results_list]), end="\n\n")



    ################################################################################################################################
    ############################# Extract User Info and Clean Scraped Data - Uses Multiprocessing ##################################

    # parallel processing to make it faster - more than 60K users
    print("Num of threads available: ", mp.cpu_count())

    start = time.time()

    if DRY_RUN:
        dry_run_index2 = 100
    else:
        dry_run_index2 = len(all_results_list)

    with mp.Pool(processes = mp.cpu_count()) as pool:
        all_extracted_values_list_mp = pool.starmap(
            extract_and_clean_info_from_html_mp,
            iterable = [
                [request_result] for request_result in all_results_list[0:dry_run_index2]
            ]
        )

    print("Total time taken for cleaning and extraction: ", time.time() - start, end="\n\n")



    ############################################################################################################################
    ###################### Convert Clean Scraped Data into DataFrame and Calculate Additional Metrics ##########################

    ### convert scraped data into dataframe
    user_info_df = pd.DataFrame(
        all_extracted_values_list_mp,
        columns = ['post_karma', 'comment_karma', 'join_date', 'gilded_posts', 'gilded_post_subreddits']
    )

    user_info_df = pd.concat(
        [users.reset_index(drop=True), user_info_df],
        axis = 1
    )

    ### convert list of subreddits into individual rows
    user_info_df_extended = user_info_df.explode(
        'gilded_post_subreddits'
    )

    ### check for nulls
    print("---"*30)
    print("User Info Data: ", end="\n\n")
    print("Null Summary: ", user_info_df_extended.isnull().sum(), end="\n\n", sep="\n")
    user_info_df_extended['gilded_post_subreddits'].fillna('None', inplace=True)


    author_gilded_subs_info = user_info_df_extended.groupby(
        ['author', 'gilded_post_subreddits']
    )[['gilded_posts']].count().reset_index()

    author_gilded_subs_info = pd.concat(
        [
            author_gilded_subs_info.rename(columns={'gilded_posts':'gilded_post_count'}),
            author_gilded_subs_info.groupby('author')['gilded_posts'].rank(method='first', ascending=False)
        ],
        axis=1
    )

    author_gilded_subs_info = author_gilded_subs_info[author_gilded_subs_info['gilded_posts'] == 1]


    ### get top gilded subreddit name for each user based on count of posts
    author_top_gilded_subreddit = author_gilded_subs_info[
        ['author', 'gilded_post_subreddits']
    ].rename(columns={'gilded_post_subreddits':'top_gilded_subreddit'})

    print("---"*30)
    print(author_top_gilded_subreddit.head(), end="\n\n")


    ### get total count of unique gilded subreddits for each user
    author_unique_gilded_subreddits = user_info_df_extended.groupby(
        ['author']
    )['gilded_post_subreddits'].nunique().reset_index()
    author_unique_gilded_subreddits = author_unique_gilded_subreddits.rename(
        columns={'gilded_post_subreddits':'gilded_unique_subs_count'}
    )

    print("---"*30)
    print(author_unique_gilded_subreddits.head(), end="\n\n")


    ### merge all user info data together
    user_info_df_final = pd.merge(
        user_info_df, 
        author_top_gilded_subreddit,
        on = 'author',
        how = 'left'
    )

    user_info_df_final = pd.merge(
        user_info_df_final, 
        author_unique_gilded_subreddits,
        on = 'author',
        how = 'left'
    )


    print("---"*30)
    print("Final User Info Data: ", end="\n\n")
    print(user_info_df_final.info(), end="\n\n", sep="\n")
    print("Null Summary: ", user_info_df_final.isnull().sum(), end="\n\n", sep="\n")
    print(user_info_df_final.head(3), end="\n\n")


    ### save final user info data as csv
    if DRY_RUN:
        print("No File Written - Dry Run")
    else:
        user_info_df_final.to_csv('data/user_data/user_info.csv', index=False)
