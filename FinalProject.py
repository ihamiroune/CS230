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

############################ Page Configurations ###############################################
st.set_page_config(
    page_title="Real-Time Data Boston Crime Dashboard",
    page_icon=":cop:",
    layout="wide",
)

############################ Global Constants ###############################################
#Define Global Constants and Variables.
BOSTON_CRIME_DATA =  "BostonCrime2022_8000_sample.CSV"
BOSTON_DISTRICTS = 'BostonDistricts_names.csv'

#### Define Color Values for each District.
# Py Deck map chart takes in RGB tuples as colors, whereas plotly_express charts take in named colors.
# Since I was not able to figure out how to dynamically get one value from the other,
# I have maintained two lists, COLOR_LIST by color name, and RGBD_LIST by decimal RBD typles.
# I had to follow a 3-step manual process to construct RGBD_LIST from COLOR_LIST.

#   a. pick a set of colors from https://matplotlib.org/3.1.1/gallery/color/named_colors.html
COLOR_LIST = ['green','orange','yellow','lightsteelblue',
              'cornflowerblue','royalblue','mediumslateblue','slateblue',
              'darkslateblue','blue','mediumblue','navy']
#   b. get the normal RGB value using: use new_list = [colors.to_rgb(c) for c in COLOR_LIST]
RGBN_LIST = [(0.0, 0.5019607843137255, 0.0), (1.0, 0.6470588235294118, 0.0), (1.0, 1.0, 0.0),
             (0.6901960784313725, 0.7686274509803922, 0.8705882352941177),(0.39215686274509803, 0.5843137254901961, 0.9294117647058824), (0.2549019607843137, 0.4117647058823529, 0.8823529411764706),
             (0.4823529411764706, 0.40784313725490196, 0.9333333333333333),(0.41568627450980394, 0.35294117647058826, 0.803921568627451), (0.2823529411764706, 0.23921568627450981, 0.5450980392156862),
             (0.0, 0.0, 1.0), (0.0, 0.0, 0.803921568627451), (0.0, 0.0, 0.5019607843137255)]
#   c. lookup the decimal RGB value from https://doc.instantreality.org/tools/color_calculator/
RGBD_LIST = [(0, 128, 0),(255, 165, 0),(255, 255, 0),
             (176, 196, 222),(100, 149, 237),(65, 105, 225),
             (123, 104, 238),(106, 90, 205),(72, 61, 139),
             (0, 0, 255),(0, 0, 205),(0, 0, 128)]


########################### Misc Utility Methods ###########################################
# DataFrame method Value_Count() returns a Series, and we often have to convert it to
# a DataFrame, before passing it to a chart function. Since we do this for several chart,
# creating a utility method that does that for code simplification.

def SeriesToDataFrame(s, ind="Index", val="Count"):
    return(pd.DataFrame({ind:s.index,val:s.values}))


########################### Read files #####################################################
def get_data(csv_list) -> pd.DataFrame:
    pd_list = [pd.read_csv(csv_file) for csv_file in csv_list]
    return tuple(pd_list)

def edit_data(df_crime, df_district):
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
    df_district["norm_color"] = RGBN_LIST
    df_district["color_name"] = COLOR_LIST

    ##### Merge district DataFrame into crime DataFrame for ease of access.
    df_crime = df_crime.merge(df_district, left_on='DISTRICT', right_on='DISTRICT_NUMBER', how='left')

    return(df_crime, df_district)


########################### Side Bar #############################################################
# Must return modified dataframe and topN value to caller, so changes made inside this method are sent back through return value.
# Since two values need to be sent back, return as a tuple.
def create_sidebar(df):
    st.sidebar.header('View Filters:')

    select_top_N = st.sidebar.slider(
        "Select Top N values to display (default is 5):",
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



######################################## Metrics ###################################################
def show_metrics(df):
    # create three columns
    kpi1, kpi2, kpi3 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Total Crimes",
        value= len(df.index)
    )

    kpi2.metric(
        label="Weekly Crime Rate",
        value= round((len(df.index))/7)
    )

    kpi3.metric(
        label="Reported Shootings",
        value = df[df['SHOOTING'] == 1].INCIDENT_NUMBER.count()
    )

###########################################   Map ###########################################

def show_map(df):
    st.markdown("### Geographical View")

    if (df.empty):
        st.write("No Data to Show. Please Select a District.")
    else:
        view_state = pdk.ViewState(
            latitude=df["lat"].mean(),
            longitude=df["lon"].mean(),
            zoom=11,
            pitch=0)

        layer = pdk.Layer('ScatterplotLayer',
                  data=df[["DISTRICT_NAME", "lat", "lon", "color"]],
                  get_position='[lon, lat]',
                  get_radius=150,
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

####################################### Crime Categories ##############################################
def show_crime_classification(df, color_map, topN):
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


####################################### Crime Patterns ##############################################
def show_crime_patterns(df, color_map):
    st.markdown("### Crime Patterns")
    fig_weekday, fig_hourly = st.columns(2)

    with fig_weekday:
        #By default, DAY_OF_WEEK is sorted as a Str. To sort as weekday (Monday to Sunday),
        #we need to convert df['DAY_OF_WEEK'] to a custom type column, and then sort it in the histogram call.
        cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        cat_type = CategoricalDtype(categories=cats, ordered=True)
        df['DAY_OF_WEEK']=df['DAY_OF_WEEK'].astype(cat_type).sort_index()  #make this column ordered, and thus sortable.
        #SOURCE: https://technology.amis.nl/data-analytics/ordering-rows-in-pandas-data-frame-and-bars-in-plotly-bar-chart-by-day-of-the-week-or-any-other-user-defined-order/
        #https://plotly.com/python/categorical-axes/
        #Source: https://plotly.github.io/plotly.py-docs/generated/plotly.express.histogram.html
        fig = px.histogram(df, x="DAY_OF_WEEK",
                           color="DISTRICT_NAME",
                           pattern_shape="SHOOTING",
                           category_orders={"DAY_OF_WEEK": cats},
                           labels ={'DAY_OF_WEEK':'Day of the Week'},
                           color_discrete_map=color_map,
                           title="By Day of the Week")
        st.write(fig)

    with fig_hourly:
        fig = px.histogram(df, x="HOUR",
                           color="DISTRICT_NAME",
                           pattern_shape="SHOOTING",
                           labels ={'HOUR':'Hour of the Day'},
                           color_discrete_map=color_map,
                           title='By Hour of the Day')
        st.write(fig)


####################################### Crime Trends ##############################################
def show_trends(df):
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



################################### Data Table ########################################################
def show_data_tables(df):
    st.markdown("### Table View")
    fig1, fig2, fig3 = st.columns([2,1,1])  #first column is twice as wide as the other two.

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



def main():
    # Setup main variables, df, dfd, and color_map. - Two DataFrames for crime data and district data.
    df, dfd = get_data([BOSTON_CRIME_DATA, BOSTON_DISTRICTS])
    df, dfd = edit_data(df, dfd)
    color_map = dict(zip(dfd["DISTRICT_NAME"].tolist(), COLOR_LIST))

    # Setup main page general attributes - Title and SideBar
    st.title("Live Boston Crime Dashboard")
    df, topN = create_sidebar(df) # create_sidebar changes the rows of DataFrame df based on the selections.

    # Call a different method for each section of the page.
    show_metrics(df) # global KIPs
    show_map(df) # city map
    show_crime_classification(df, color_map, topN)
    show_crime_patterns(df, color_map )
    show_trends(df)
    show_data_tables(df)

main()

