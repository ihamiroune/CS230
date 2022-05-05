"""
CS 230 Final Program
Boston Crimes Data:  Crime Data for the City of Boston in 2022

Ihsan Hamiroune
"""
import numpy as np  # np mean, np random
import plotly.express as px  # all interactive charts, except map
import streamlit as st  # 🎈 data web app development
import pydeck as pdk #used for map chart
import pandas as pd  # read csv, df/series manipulation
from pandas.api.types import CategoricalDtype  # for day_of_week custom type
from pathlib import Path

############################ Page Configurations ###############################################
st.set_page_config(
    page_title="Real-Time Data Boston Crime Dashboard",
    page_icon=":cop:",
    layout="wide",
)

############################ Global Constants #####################################################
#Define Global Constants and Variables.
BOSTON_CRIME_DATA = Path(__file__).parents[0]/'Data/BostonCrime2022_8000_sample.csv'
BOSTON_DISTRICTS = Path(__file__).parents[0]/'Data/BostonDistricts_names.csv'

########################### Define Color Values for each District ##################################
# Py Deck map chart takes in RGB tuples as colors, whereas plotly_express charts take in named colors.
# Since I was not able to figure out how to dynamically get one value from the other,
# I have maintained two lists, COLOR_LIST by color name, and RGBD_LIST by decimal RBD typles.
# I had to follow a 3-step manual process to construct RGBD_LIST from COLOR_LIST.

#   a. pick a set of colors from https://matplotlib.org/3.1.1/gallery/color/named_colors.html
COLOR_LIST = ['green','orange','yellow','lightsteelblue',
              'cornflowerblue','royalblue','mediumslateblue','slateblue',
              'darkslateblue','blue','mediumblue','navy']
#   b. lookup the decimal RGB value from https://doc.instantreality.org/tools/color_calculator/
RGBD_LIST = [(0, 128, 0),(255, 165, 0),(255, 255, 0),
             (176, 196, 222),(100, 149, 237),(65, 105, 225),
             (123, 104, 238),(106, 90, 205),(72, 61, 139),
             (0, 0, 255),(0, 0, 205),(0, 0, 128)]


################################ Misc Utility Methods ###########################################
# DataFrame method Value_Count() returns a Series, and we often have to convert it to
# a DataFrame, before passing it to a chart function. Since we do this for several chart,
# creating a utility method that does that for code simplification.
def SeriesToDataFrame(s, ind="Index", val="Count"):
    """
    parameter(s): series, index name, value
    return: data frame with two columns, index and count
    """
    return(pd.DataFrame({ind:s.index,val:s.values}))

##################################### Read files #####################################################
def get_data(csv_list) -> pd.DataFrame:
    """
    parameter(s): imported csv list
    return: tuple of dataframes that are made from csv list items
    """
    pd_list = [pd.read_csv(csv_file) for csv_file in csv_list]
    return tuple(pd_list)

def edit_data(df_crime, df_district):
    """
    parameter(s): two dataframes generated from get_data() 
    return: cleaned up data frames (remove incomplete records, add new columns, rename columns)
    """
    ##### Cleanup df_crime data
    #Streamlit: Map data must contain a column named "latitude" or "lat"
    df_crime.rename(columns = {'Lat' : 'lat', 'Long' : 'lon'}, inplace = True)

    #exclude rows with invalid/incomplete data
    df_crime = df_crime[pd.notna((df_crime['DISTRICT']))]  # exclude records with Null District.
    df_crime = df_crime[(df_crime['lat'] != 0) &  (df_crime['lon'] != 0 ) & (df_crime['DISTRICT'] != 'External')]

    #construct date column from <date time> Column.
    df_crime['date']= df_crime['OCCURRED_ON_DATE'].str.split(" ", expand=True)[0]

    ##### Define district specific attributes
    df_district["color"] = RGBD_LIST
    #df_district["norm_color"] = RGBN_LIST
    df_district["color_name"] = COLOR_LIST

    ##### Merge district DataFrame into crime DataFrame to capture district name and color values.
    df_crime = df_crime.merge(df_district, left_on='DISTRICT', right_on='DISTRICT_NUMBER', how='left')

    return(df_crime, df_district)


################################### Side Bar ###################################################
#Side bar: Must return modified dataframe and topN value to caller, so changes made inside this method are sent back through return value.
#Since two values need to be sent back, return as a tuple.

def create_sidebar(df):
    """
    parameter(s): crime dataframe
    return: modified data frame based on sidebar filters, and top N selection
    """

    st.sidebar.header('View Filters:')

    select_top_N = st.sidebar.slider(
        "Select Top Number of Categories to display (default is 5):",
        5, 15, 5)

    select_district = st.sidebar.multiselect(
        "Select Districts to include (default is all):",
        options = df["DISTRICT_NAME"].unique(),
        default = df["DISTRICT_NAME"].unique()
    )

    select_week = st.sidebar.multiselect(
        "Select Days of the Week to view (default is all)",
        options = df["DAY_OF_WEEK"].unique(),
        default = df["DAY_OF_WEEK"].unique()
    )

    select_shooting = st.sidebar.checkbox(
        'View Shootings Only.')

    st.sidebar.markdown("### Breaking News")
    st.sidebar.video("https://www.youtube.com/watch?v=9u51XhUYT_E")

    if select_shooting:
        df = df[df['DAY_OF_WEEK'].isin(select_week) & df['DISTRICT_NAME'].isin(select_district) & df['SHOOTING'] == 1]
    else:
        df = df[df['DAY_OF_WEEK'].isin(select_week) & df['DISTRICT_NAME'].isin(select_district)]
    return(df, select_top_N)



######################################### Metrics ###################################################
#Metric: Creating page metrics (Total Crimes, Weekly Crimes, and Reported Shootings) based on crime dataframe. 

def show_metrics(df):
    """
    parameter(s): crime dataframe
    return: none
    """

    # create three columns
    kpi1, kpi2, kpi3 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Total Crimes",
        value= len(df.index)
    )

    kpi2.metric(
        label="Weekly Crimes",
        value= round((len(df.index))/7)
    )

    kpi3.metric(
        label="Reported Shootings",
        value = df[df['SHOOTING'] == 1].INCIDENT_NUMBER.count()
    )

#############################################   Map  ################################################
#Map: Creating a map based on crime dataframe, showing crimes/shootings per district. Map will end up being 
#color-coded by district

def show_map(df):
    """
    parameter(s): crime dataframe
    return: none
    """

    st.markdown("### Geographical View")

    if (df.empty):
        st.write("No Data to Show. Please Select a District.")
    else:
        view_state = pdk.ViewState(
            latitude=df["lat"].mean(),
            longitude=df["lon"].mean(),
            zoom=10,
            pitch=0)

        layer = pdk.Layer('ScatterplotLayer',
                  data=df[["DISTRICT_NAME", "lat", "lon", "color"]],
                  get_position='[lon, lat]',
                  get_radius=100,
                  get_color='color',
                  pickable=True
                  )

        tool_tip = {"html": "{DISTRICT_NAME}",
            "style": { "backgroundColor": "steelblue",
                  "color": "white"}
         }


        map = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v10',
            initial_view_state=view_state,
            layers=[layer],
            tooltip= tool_tip
        )

        st.pydeck_chart(map)

########################################### Crime Categories ##############################################
#Crime categories: Display most dangerous districts and most common crimes based on crime dataframe

def show_crime_classification(df, color_map, topN):
    """
    parameter(s): crime dataframe, color selection based on district, top N selection from sidebar 
    return: none
    """

    st.markdown("### Crime Classification")
    fig_district, fig_offense = st.columns(2)


    with fig_district:
        df_districts = df['DISTRICT_NAME'].value_counts()[:topN]

        fig = px.pie(df, names=df_districts.index,
                     color=df_districts.index, #Must be defined for color_discrete_map to take effect
                     values=df_districts.values,
                     title='Most Dangerous Districts',
                     color_discrete_map=color_map)
        st.write(fig)

    with fig_offense:
        df_top_offenses = df['OFFENSE_DESCRIPTION'].value_counts()[:topN]
        fig = px.pie(df, names=df_top_offenses.index,
                     values=df_top_offenses.values,
                     title='Most Common Offenses',
                     color_discrete_sequence=px.colors.sequential.Bluyl)
        st.write(fig)


########################################## Crime Patterns ##############################################
#Crime Patterns: Display crime patterns by day of the week, and hour of the day. Data from crime dataframe.

def show_crime_patterns(df, color_map):
    """
    parameter(s): crime dataframe, color selection based on district 
    return: none
    """
    st.markdown("### Crime Patterns")
    fig_weekday, fig_hourly = st.columns(2)

    with fig_weekday:
        #By default, DAY_OF_WEEK is sorted as a Str. To sort as weekday (Monday to Sunday),
        #we need to convert df['DAY_OF_WEEK'] to a custom type column, and then sort it in the histogram call.

        days = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_type = CategoricalDtype(categories=days, ordered=True)
        df['DAY_OF_WEEK']=df['DAY_OF_WEEK'].astype(day_type).sort_index()  #make this column ordered, and thus sortable.

      
        
        fig = px.histogram(df, x="DAY_OF_WEEK",
                           color="DISTRICT_NAME",
                           category_orders={"DAY_OF_WEEK": days},
                           labels ={'DAY_OF_WEEK':'Day of the Week'},
                           color_discrete_map=color_map,
                           title="By Day of the Week")
        st.write(fig)

    with fig_hourly:
        fig = px.histogram(df, x="HOUR",
                           color="DISTRICT_NAME",
                           labels ={'HOUR':'Hour of the Day'},
                           color_discrete_map=color_map,
                           title='By Hour of the Day')
        st.write(fig)


####################################### Crime Trends ##############################################
#Crime Trends: Display daily crime rate and cumulative crime rate by date 
 
def show_trends(df):
    """
    parameter(s): crime dataframe 
    return: none
    """
    st.markdown("### Crime Trends")

    df_date_count = SeriesToDataFrame(df['date'].value_counts().sort_index(),"date", "count")
    df_date_count['cum_count'] = np.cumsum(df_date_count['count'])

    fig_rate, fig_cumulative = st.columns(2)
    with fig_rate:
        area_graph = px.area(df_date_count, x="date", y="count",
                             color_discrete_sequence=px.colors.sequential.Bluered,
                             title='Daily Rate')
        st.write(area_graph)

    with fig_cumulative:
        area_graph = px.area(df_date_count, x="date", y="cum_count",
                           color_discrete_sequence=px.colors.sequential.Bluered,
                           title='Cumulative Counts')
        st.write(area_graph)



########################################## Data Table ###################################################
#Data Table: Present figure data in table form 
 
def show_data_tables(df):
    """
    parameter(s): crime dataframe 
    return: none
    """
    st.markdown("### Table View")
    fig1, fig2, fig3 = st.columns(3)  

    with fig1:
        st.write("Crime Counts by Offense.")
        df_byoffense = SeriesToDataFrame(df['OFFENSE_DESCRIPTION'].value_counts(),
                                         "Offense", "Count").style.set_properties(**{'text-align': 'left'})
        st.dataframe(df_byoffense)

    with fig2:
        st.write("Crime Counts by District.")
        df_bydistrict = SeriesToDataFrame(df['DISTRICT_NAME'].value_counts(),
                                          "District", "Count").style.set_properties(**{'text-align': 'left'})
        st.dataframe(df_bydistrict)

    with fig3:
        st.write("Crime Counts by Street.")
        df_bystreet = SeriesToDataFrame(df['STREET'].value_counts(),
                                        "Street", "Count").style.set_properties(**{'text-align': 'left'})
        st.dataframe(df_bystreet)


################################################# Main ######################################################

def main():
    # Setup main variables, df, dfd, and color_map. - Two DataFrames for crime data and district data.
    df, dfd = get_data([BOSTON_CRIME_DATA, BOSTON_DISTRICTS])
    df, dfd = edit_data(df, dfd)
    color_map = dict(zip(dfd["DISTRICT_NAME"].tolist(), COLOR_LIST))  #plotly expects color info in dictionary format, not list format.

    # Setup main page general attributes - Title and SideBar
    st.title("2022 Boston Crime Dashboard")
    df, topN = create_sidebar(df) # create_sidebar changes the rows of DataFrame df based on the selections.

    # Call a different method for each section of the page.
    show_metrics(df) # global KIPs
    show_map(df) # city map
    show_crime_classification(df, color_map, topN)
    show_crime_patterns(df, color_map )
    show_trends(df)
    show_data_tables(df)

main()

################################################## Sources ####################################################################
#SOURCE: https://technology.amis.nl/data-analytics/ordering-rows-in-pandas-data-frame-and-bars-in-plotly-bar-chart-by-day-of-the-week-or-any-other-user-defined-order/
#https://plotly.com/python/categorical-axes/
#Source: https://plotly.github.io/plotly.py-docs/generated/plotly.express.histogram.html
