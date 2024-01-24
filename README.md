# 381E - Data Science

### Lecturer: [Prof. Dr. Atabey Kaygun](https://web.itu.edu.tr/kaygun/)

This repository includes all my practical work for the Introduction to Data Science class which I participated during Fall_23 term at my university, Istanbul Technical. All my work is graded by the lecturer himself and his feedbacks can be seen here. 

# About the Class                                        
                                       
The lecture covered key concepts of data science starting with data cleaning, feature engineering and data analysis. Foundational concepts for importing, cleaning, transforming, visualizing, and extracting insights from complex datasets were introduced, and we were given many practical tasks to have experience with real-world data. All the programming was done using Python, and Jupyter Notebooks was used as the development environment. NumPy, Pandas, SciPy, Matplotlib, and Vega were among the common libraries utilised during the course. 

Another focus of the lecture was text procesing. It covered NLP techniques such as extracting summaries and creating text embeddings with models like Bert and Word2Vec. Modeling using vectorized texts were shown, and we were given mini projects on it. 

This class was really extensive, the topics it tought ranged from dimensionality reduction to working with audio data. However, I believe that the most beneficial part was the mini projects we needed to finish throughout the term. I've spent quite some time dealing with them and while doing so, I learned a lot more than I expected. The official page of the class can be found [here](https://github.com/kaygun/2023-Fall-381E).

# Mini Projects

Here I will write down the instructions for the mini project homeworks which I completed during the term. All my work was graded on this repository by the lecturer, and feedbacks can be seen on top of the notebooks. How I managed to perform these tasks below was openly explained on each notebook I have in my repo. The course allowed and even encouraged us to use large language models such as ChatGPT, Claude 2, LLAMA, Mistral etc. during our learning. In my opinion, this was really beneficial for our learning process.  

## Homework #1

Overall rules:

- Refrain from downloading and loading data from a local file. Instead, obtain all required data using the appropriate API.
- Do not download data in separate parts. For instance, in Q1 where data for 22 countries is necessary, obtain the data in a single, large piece and partition it later on.
- Avoid duplicating code by copying and pasting it from one cell to another. If copying and pasting is necessary, develop a suitable function for the task at hand and call that function.
- When providing parameters to a function, never utilize global variables. Instead, always pass parameters explicitly and always make use of local variables.

Failure to adhere to these guidelines will result in a 25-point deduction for each infraction.

### Q1

For this question, you are going to use [World Bank's Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/).

The Mediterranean Sea is bordered by following 22 countries: Spain, France, Monaco, Italy, Slovenia, Croatia, Bosnia and Herzegovina, Montenegro, Albania, Greece, Turkey, Syria, Lebanon, Israel, Palestine, Egypt, Libya, Tunisia, Algeria, Morocco, Malta, and Cyprus. Using the World Bank's Climate Change Knowledge Portal, obtain the following data through their API:

- Monthly average surface temperatures
- Monthly average precipitation

Gather this information as a time series from 1950 to 2020 for all countries surrounding the Mediterranean Sea. Conduct the following analyses on the retrieved data:

1. Compute the average surface temperature per month for each country, save the results in a pandas data frame called `MAST`, and display the data using a line plot.
2. Compute the average precipitation per month for each country, save the results in a pandas data frame called `MAPR`, and display the data using a line plot.
3. Compute the average surface temperature per annum for each country, save the results in a pandas data frame called `YAST`, and display the data using a line plot.
4. Compute the average precipitation per annum for each country, save the results in a pandas data frame called `YAPR`, and display the data using a line plot.


### Q2

For this question, you are going to use [yfinance](https://pypi.org/project/yfinance/) library to access [Yahoo! Finance Data](https://finance.yahoo.com/).

Using the `yfinance` library, obtain the stock prices of the following companies between January 1st, 2019 and December 31st, 2022: Alphabet (Google), Meta (Facebook), Cisco, Intel, AMD, Qualcomm, Apple, Samsung, Xiaomi, and Tesla.

1. Construct a pandas data frame `CHANGE` containing values of $-1$, $0$, or $1$ for each trading day within the specified time frame for each mentioned company. Assign a value of $0$ if the daily closing price change is within $\pm 2.5\%$ of the opening price. If the change is greater than $2.5\%$ of the opening price, assign a value of $1$. If the change is less than $-2.5\%$, assign a value of $-1$.
2. Identify the longest continuous time intervals during which the `CHANGE` consistently remains $1$ and $-1$ for each company.
3. Create another pandas data frame `DIFFERENCE` consisting of values $-1$, $0$, or $1$ for each day within the specified time period (excluding January 1st, 2019). If the opening price of a day is within $\pm 2.5\%$ of the previous day's opening price, assign a value of $0$. If the change in the opening price in comparison to the previous day is greater than $2.5\%$, assign a value of $1$. If the change in the opening price is less than $-2.5\%$, assign a value of $-1$.
4. Determine the longest continuous time intervals during which the `DIFFERENCE` consistently remains $1$ and $-1$ for each company.



## Homework #2

Overall rules:

- Refrain from downloading and loading data from a local file. Instead, obtain all required data using the appropriate API. This rule does not apply to the TUIK data.
- Refrain from cleaning data by hand on a spreadsheet. All cleaning must be done programmatically, with each step explained. This is so that I can replicate the procedure deterministically.
- Refrain from using code comments to explain what has been done. Document your steps by writing appropriate markdown cells in your notebook.
- Avoid duplicating code by copying and pasting it from one cell to another. If copying and pasting is necessary, develop a suitable function for the task at hand and call that function.
- When providing parameters to a function, never use global variables. Instead, always pass parameters explicitly and always make use of local variables.
- Document your use of LLM models (ChatGPT, Claude, Code Pilot etc). Either take screenshots of your steps and include them with this notebook, or give me a full log (both questions and answers) in a markdown file named `HW2-LLM-LOG.md`.

Failure to adhere to these guidelines will result in a 15-point deduction for each infraction.


### Q1

For this question, we are going to use [Istanbul Municipality Data Service](https://data.ibb.gov.tr/). Specifically, we are going to look at the 'Saatlik Trafik Yoğunluk Veri Seti' dataset.

1. Write a function that takes a year and a month as parameters that pulls the data for that specific year and month and returns the data as a pandas dataframe.
2. Using this function get the data for September 2023.
3. Group the data with respect to GEOHASH column, and then aggragate
   - 'LONGITUDE' column using 'first'
   - 'LATITUDE' column using 'first'
   - 'AVERAGE_SPEED' using 'mean'
   - 'NUMBER_OF_VEHICLES' using 'mean'
4. Find the first 25 data points with
   - the highest average number of vehicles
   - the highest average speed
   - the lowest average speed
5. Create a heatmap using [folium](https://pypi.org/project/folium/) library on the 'NUMBER_OF_VEHICLES' column of the dataframe you constructed in Step 2.

### Q2

For this question, we are going to build a map. To build a map, we are going to merge several data sources.

- Get the shape files for Turkish administrative level 0 (iller) and level 1 (ilçeler) from [GADM](https://gadm.org) using their API.
- Get the census population data (both at level 0 and level 1) for the year 2000 from [TUIK](https://biruni.tuik.gov.tr) using their API.
- Get the crime statistics (suç türü ve suçun işlendiği ile göre infaz kurumuna giren hükümlüler) from [TUIK](https://biruni.tuik.gov.tr) using their API.

You need to poke around GADM and TUIK to find out which data you will need. For the GADM data, the API allows URL access. For TUIK data, you will have to download the data into a local file and load it. However, DO NOT clean data by hand. The raw data has to be loaded as a pandas frame and cleaning must be done programmatically for replication purposes. You will have to push the files you downloaded to github. Name the files as `HW2-Population-level0.xlsx`,  `HW2-Population-level1.xlsx` and `HW2-Crime.xlsx`.

After getting the data

1. Create a [GeoPandas][1] dataframe from the shape data, and merge it with population data (with level 1).
2. Using GeoPandas sketch a choropleth map for the population at administrative level 1 (ilçeler)
3. Merge shape data (level 0), the population data (level 0) and the cleaned crime data.
4. On the dataframe you created in Step 3, transform the totals for each crime type for each municipality into a percentage by dividing it by the correct population number.
5. Using the data you generated in Step 4, sketch two choropleth maps using GeoPandas for the two crime types of your choosing at administrative level 0 (iller)

[1]: https://geopandas.org/en/stable/


### Q3

For this question, you are going to use [WorldBank Data API](https://data.worldbank.org/) through the python library [wbgapi](https://pypi.org/project/wbgapi/). 

1. Get the following data for the following countries: Albania, Bosnia and Herzegovina, Bulgaria, Croatia, Kosovo, Montenegro, North Macedonia, Romania, Serbia, Slovenia, Turkey, Greece, Cyprus, Malta, Italy, Poland, Belarus, and Ukraine.

* Child mortality under 5 year olds
* Female literacy rate for ages 15+
* Female labor force participation rate for ages 15+) 
* Poverty headcount ratio at $3.65 a day as a percentage of the population
* GDP Per capita
* CPIA gender equality rating

2. Merge the data you obtained into a single data frame such that you have the following columns. Pivot the data frames whenever necessary:

- Year
- Country
- Child Mortality
- Female Literacy
- Female Workforce Participation
- Poverty
- GDP
- Gender Equality

3. Write a function that takes the following as parameters

- The data frame
- The name of the country
- The column name

that plots the time series (in years, in the correct order) of the corresponding column for the corresponding country. Sketch 4 such graphs of your choosing, both in terms of country and column.

4. Write a function that takes the following as parameters

- The data frame
- Years as a list 
- A pair of column names

that sketches a scatter plot of the pair of columns for every country for the list of particular years. Sketch 4 such graphs of your choosing from 2010 to 2020.

5. Sketch 'Child Mortality' against 'Female Literacy' for all countries and for all available years as a scatter plot. Analyze the graph and draw conclusions.

6. Sketch 'Female Workforce Participation' against 'Child Mortality' for all countries and for all available years as a scatter plot. Analyze the graph and draw conclusions.
   
7. Make a violin plot of 'Female Literacy' column against 'Gender Equality' column for all countries and for all available years. Investigate any correlation between these variables.

## Homework #3

Overall rules:

- Refrain from using code comments to explain what has been done. Document your steps by writing appropriate markdown cells in your notebook.
- Avoid duplicating code. Do not copy and paste code from one cell to another. If copying and pasting is necessary, write a suitable function for the task at hand and call that function.
- When providing parameters to a function, never use global variables. Instead, always pass parameters explicitly and always make use of local variables.
- Document your use of LLM models (ChatGPT, Claude, Code Pilot etc). Either take screenshots of your steps and include them with this notebook, or give me a full log (both questions and answers) in a markdown file named `HW3-LLM-LOG.md`.

Failure to adhere to these guidelines will result in a 25-point deduction for each infraction.

### The Dataset

For this homework, we are going to use the [data warehouse](https://clerk.house.gov/Votes/) for the [US House of Representatives](https://www.house.gov/). The data server has data on each vote going back to 1990. The voting information is in XML format. For example, the code below pulls the data for the 2nd roll call from 1990 Congress.


### Pull all the data from 1990 to 2023, and store it for questions below.

### Q1

1. Not all of the roll calls are votes. For example, some roll calls are QUORUMs (yoklama). For each year, find out the legislator and his/her state who were absent the most.
2. For each year and for each state find out how many legislators there are. For example, in 1990 California had 45 legislators while Vermont had 1.
3. Create a data frame with the following columns:
   - Year
   - State
   - Name of the Legislator
   - His/her party affiliation (Democrat/Republican/Independent)
   - Number of times he/she voted
   - Number of times he/she did not vote
5. Find out who is the longest serving legislator in the US House representative.

### Q2

For this question, we are going to measure polarization in the US Congress.

For this specific vote example above, the YEAS and NAYS are tabulated as follows:


|             |  YEAs   |  NAYs  |
|-------------|---------|--------|
| Democrats   |    234  |     2  |
| Republicans |     78  |    87  |


We are going to measure **polarization** by the following formula

$$ \frac{|\text{Difference in YEAs}| + |\text{Difference in NAYs}|}{\text{Total number of votes}} $$

For this particular vote the polarization is

$$ \frac{|234-78|+|2-87|}{234+78+2+87} \approx 0.6 $$

1. Measure polarization for each roll call and store it in a data frame with the date information.
2. Plot the results against time.
3. Analyze the results. Did polarization increase, decrease, or stayed the same?

### Q3

For this question, we are going to measure if each legislator voted along his/her part or voted against the party lines. For example, in the example above there are two Democratic legislators broke the party line and voted NAY while 234 other Democrats voted YEA. Those legislators were Jacobs from Indiana and Schroeder from Colorado.

1. For each legislator and for each year, find out the number of times they voted in total.
2. For each legislator and for each year, find out the number of times they voted along the party lines, and the number of times they broke the party line.
3. For each year and for each party, count the number of legislators that never broke the party line in that year.
4. For each year, list the top 5 legislators (and their party affiliation) that broke the party line the most.

### Q4

For this question, we are going to look at the text of each vote question. For example, the vote question for the example roll call is 'On Approving the Journal'. This is an open-ended question, and you must design an experiment and choose a specific machine learning algorithm to find out the answer if there is one.

1. For each party, find out if there are specific issues that they prefer voting 'YEA' or 'NAY'. For example, it is widely believed that Democrats vote 'NAY' on issues restricting abortion while Republicans vote 'YEAH'. For this question you are looking for a quantifiable connection between the text of the vote question and the likelihood of each party voting 'YEA' or 'NAY'.

2. Now, do the same for each legislator to find out the issues each legislator cares about in each year.

## Homework #4

### Overall rules:

- Refrain from saving datasets locally. You may experiment with your answers on a locally saved version of the datasets, but do not upload your local files with your homework as the datasets are very large. In your submitted answers datasets should be read from the original source URL.
- Document all of your steps by writing appropriate markdown cells in your notebook. Refrain from using code comments to explain what has been done. 
- Avoid duplicating code. Do not copy and paste code from one cell to another. If copying and pasting is necessary, write a suitable function for the task at hand and call that function.
- Document your use of LLM models (ChatGPT, Claude, Code Pilot etc). Either take screenshots of your steps and include them with this notebook, or give me a full log (both questions and answers) in a markdown file named `HW4-LLM-LOG.md`.

Failure to adhere to these guidelines will result in a 25-point deduction for each infraction.

### Q1 (Author attribution)

For this question we are going to use two novels:
- [Jayne Eyre](https://www.gutenberg.org/ebooks/1260) by Charlotte Bronte, and
- [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) by Jane Austen.

1. Get the novels' plain text versions and remove all the parts which do not belong to the novels.
2. Using NLTK's sentence tokenizers, tokenize each novel.
3. Label sentences by 0 or 1 depending on whether the sentence is written by Austen or Bronte, and then merge the two sentence data sets.
4. Vectorize the merged sentence dataset. (Tell the vectorizer to remove all stop words.)
5. Split the vectorized sentences and labels as train and test. Use the 25% of the data as test.
6. Train a logistic regression model on the train subset.
7. Using the model you trained, predict labels on the test set, and then construct a confusion matrix.
8. Can your model distinguish sentences written by Bronte or Austen? Analyze.


### Q2 (Voice recognition)

For this question we are going to use the [Axiom voice recognition dataset](https://zenodo.org/records/1218979). The voice data contains data for 73 individuals. Since the dataset is fairly large, do not commit your local copy to github.

1. Build machine learning models that distinguish these individuals.
2. Test your model(s) and calculate their accuracies.
3. Construct and display confusion matrix or matrices.

### Q3 (City hopping)

For this question we are going to use [Open Flight Data](https://openflights.org/data.php).

1. Pull the data on airports and routes. Clean it and put the correct names on columns. Read the [documentation on the data](https://openflights.org/data.php).
2. Merge the route dataset and the airport dataset so the combined version has route source and target airports has full airport names and cities instead of airport codes.
3. Construct a graph where each node is a city, and two cities are connected by an edge if there is a flight between these cities.
4. Find the flight routes with the minimal number of stops for the following source and target cities. For example, a route that has the minimal number of connections/stops from Antalya to Deer Lake (Canada) is Antalya, London, Halifax, Deer Lake.
   - From Adana (Turkey) to Auckland (New Zealand)
   - From Ankara (Turkey) to Kona (Hawaii, USA)
   - From Sydney (Australia) to Churchhill (Canada)
5. Using the networkx's implementation of the [Page Rank Algorithm](https://en.wikipedia.org/wiki/PageRank) find the top 10 cities that are important for the global flight network.

### Q4 ([Six degrees of Kevin Bacon](https://en.wikipedia.org/wiki/Six_Degrees_of_Kevin_Bacon))

For this question we are going to use [IMDB datasets](https://datasets.imdbws.com/). Read the [documentation](https://developer.imdb.com/non-commercial-datasets/) for what each of the dataset is for. Pull the data and merge the appropriate datasets for the followwing tasks. The datasets are fairly large. Do not commit the datasets to the github repo!

1. Construct a graph where the nodes are actors and actresses, and two actors/actresses are connected by a weighted edge if these people appeared in the same movie together. The weight of any edge is the number of movies these people appear together. For example Haluk Bilginer and Bekir Aksoy appeared together in 40 movies or TV series episodes.
2. Using the graph you constructed find the shortest paths between following pairs of actors/actresses. For example the shortest path between Haluk Bilginer and Kevin Bacon is Haluk Bilginer, Danny Glover, Patricia Clarkson, and Kevin Bacon where any two consecutive actors/actresses appeared together in a movie.
   - Charles Chaplin and Nur Sürer
   - Paul Newman and Keanu Reeves
   - Harold Lloyd and Zoey Kazan
   - Vanessa Redgrave and Tilda Swinton
3. Using eigen-value centrality, find 10 actors/actresses that are important for the actor/actress network.


### Q5 (Shopping for best universities for math PhD)

For this question is about [Math Genealogy](https://genealogy.math.ndsu.nodak.edu/index.php) and we are going to use [the data](https://github.com/j2kun/math-genealogy-scraper) that was already scraped by [Jeremy Kun](https://jeremykun.com/). We don't need the whole repo by Kun, just 'data.json' from the repository. Pull the data and process it accordingly for the tasks below. Again, since the dataset is fairly large do not commit it to your github repo.

Data consists of historical data on PhD dissertation. A sample entry would look like as follows:

    {
      "id": 13325,
      "name": "Daniel Gray Quillen",
      "thesis": "Formal Properties of Over-Determined Systems of Linear Partial Differential Equations",
      "school": "Harvard University",
      "country": "UnitedStates",
      "year": 1964,
      "subject": "19—K-theory",
      "advisors": [
        7583
      ],
      "students": [
        62762,
        67441,
        30219,
        62770,
        62764,
        62771
      ]
    },

The name of the person who wrote the thesis is Daniel Gray Quillen, and his unique id is 13325. His thesis title is "Formal Properties of Over-Determined Systems of Linear Partial Differential Equations". He got his PhD degree from Harvard in 1964. His advisor's id is 7583 (Raul Bott). Note that the advisors field is an array since a student might have more than 1 advisor. The student field indicates Quillen had 6 PhD students with those specific id's

1. Construct a graph where nodes are countries. Two countries A and B are connected by an edge if a mathematician whose PhD degree awared by a university in A, had a PhD student at a university in the country B, or vice versa. Attach a weight to the edge by counting the number of all such instances. For example, Quillen had his degree from Harvard which is in the US. 5 of his students got their degrees at MIT which is also in the US while the 6th got his degree from Oxford which is in the UK. Thus Quillen adds 5 to the edge (which indeed is a loop) between the US and the US, and 1 to the edge between the US and the UK.
2. Using the Page Rank algorithm find the top 10 important countries in mathematics.
3. Construct a new graph where nodes are universities with the same rules as above. Again, using the Quillen example above: Quillen adds 5 to the edge between Harvard and MIT, and 1 to the edge between Harvard and Oxford.
4. Using the Page Rank algorithm find the top 10 important universities in mathematics.
5. Now, write a function that takes the name of a math subject (such as 'K-theory') and returns the top 10 important schools for the university network properly filtered. In other words, this time you will only consider the PhD degrees in the given subject alone. Find out the top 10 universities for the following subjects:
   - Statistics
   - Group theory
   - Topology
   - Functional Analysis
