import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

app = dash.Dash(__name__)

dat = pd.read_csv("Telco_Customer.csv")


dat['TotalCharges'] = pd.to_numeric(dat['TotalCharges'], errors='coerce')
dat.dropna(subset=['TotalCharges'], inplace=True)
dat['Churn'] = dat['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
dat['AvgMonthlyCharges'] = dat['TotalCharges'] / dat['tenure']
service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
dat['TotalServices'] = dat[service_columns].apply(lambda x: sum(x == 'Yes'), axis=1)

gender_data = dat.groupby('gender').agg(
    AvgTotalCharges=pd.NamedAgg(column='TotalCharges', aggfunc='mean'),
    AvgTotalServices=pd.NamedAgg(column='TotalServices', aggfunc='mean'),
    AvgMonthlyCharges=pd.NamedAgg(column='AvgMonthlyCharges', aggfunc='mean')
).reset_index()

grouped_senior_data = senior_citizens_data.groupby('gender').agg(
    AvgTotalCharges=pd.NamedAgg(column='TotalCharges', aggfunc='mean'),
    AvgTotalServices=pd.NamedAgg(column='TotalServices', aggfunc='mean'),
    AvgMonthlyCharges=pd.NamedAgg(column='AvgMonthlyCharges', aggfunc='mean'),
    Count=pd.NamedAgg(column='gender', aggfunc='count')
).reset_index()

churn_rate = dat.groupby('tenure')['Churn'].mean()
churn_by_contract = dat.groupby('Contract')['Churn'].mean()
service_distribution = dat['InternetService'].value_counts()

features = dat[['tenure', 'MonthlyCharges', 'TotalCharges']].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
dat['Cluster'] = kmeans.labels_

dat = dat.dropna(subset=['TotalCharges'])
X = dat[['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices']]
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
y = dat['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

kmf = KaplanMeierFitter()
T = dat['tenure'] 
E = dat['Churn']  
kmf.fit(T, event_observed=E)

app.layout = html.Div(children=[
    html.H1(children='Customer Analytics Dashboard'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='churn-rate-line-chart',
        figure=px.line(dat.groupby('tenure')['Churn'].mean().reset_index(), x='tenure', y='Churn', title='Churn Rate Over Time')
    ),

    dcc.Graph(
        id='churn-by-contract-bar-chart',
        figure=px.bar(dat.groupby('Contract')['Churn'].mean().reset_index(), x='Contract', y='Churn', title='Churn by Contract Type')
    ),

    dcc.Graph(
    id='service-distribution-pie-chart',
    figure=px.pie(
        values=dat['InternetService'].value_counts().values,
        names=dat['InternetService'].value_counts().index,
        title='Customer Distribution by Internet Service'
    )
),

    dcc.Graph(
        id='customer-segments-scatter-plot',
        figure=px.scatter(dat, x='MonthlyCharges', y='TotalCharges', color='Cluster', title='Customer Segments', color_continuous_scale='Viridis')
    ),

    html.Div([
        html.H3("Confusion Matrix"),
        html.Pre(str(confusion_matrix(y_test, y_pred)))
    ]),

    html.Div([
        html.H3("Classification Report"),
        html.Pre(str(classification_report(y_test, y_pred)))
    ]),

    html.Div([
        dcc.Graph(
            id='survival-function',
            figure=px.line(x=kmf.survival_function_.index, y=kmf.survival_function_['KM_estimate'], labels={'x': 'Tenure', 'y': 'Survival Probability'}, title='Customer Survival Function')
        )
    ]),

    html.Div([
        html.H3("Gender Data Table"),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in gender_data.columns],
            data=gender_data.to_dict('records'),
        )
    ]),

    html.Div([
        html.H3("Senior Citizens Data Table"),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in grouped_senior_data.columns],
            data=grouped_senior_data.to_dict('records'),
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
