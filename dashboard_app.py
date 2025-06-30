#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: carlnordahl

Description: Script for running the dashboard visualizing data statistics and model performance.
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# paths for assets
ASSETS_DIR = 'assets'
MODELS_DIR = 'models'
DF_FINAL_PATH = os.path.join(ASSETS_DIR, 'processed_data.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_logistic_regression_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'standard_scaler.pkl')

SELECTED_FEATURES = [
    'Surprise scaled',
    'Surprise Direction',
    'Surprise_scaled_x_Volatility',
    'Market Return',
    'Volatility_x_Market_Return',
    'Relative Surprise',
    'Reported EPS',
    'Market_Return_sq'
]
TARGET_COLUMN = 'Up/Down'
RANDOM_STATE = 10

# Define colors based on Tailwind CSS palette
COLORS = {
    'emerald': '#10B981',
    'red': '#EF4444',
    'blue': '#3B82F6',
    'neutral_gray': '#6B7280' 
}


df = pd.read_csv(DF_FINAL_PATH)
df[TARGET_COLUMN] = df['Return'].apply(lambda r: 'up' if r >= 0 else 'down')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Dashboard: Data, Model, and Scaler loaded successfully.")

X_for_eval = df[SELECTED_FEATURES]
y_for_eval = df[TARGET_COLUMN]

for col in X_for_eval.columns:
    if X_for_eval[col].dtype in ['float64', 'int64'] and X_for_eval[col].isnull().any():
        X_for_eval[col] = X_for_eval[col].fillna(X_for_eval[col].mean())

_, X_test_eval, _, y_test_eval = train_test_split(
    X_for_eval, y_for_eval, test_size=0.25, random_state=RANDOM_STATE
)
X_test_scaled_eval = scaler.transform(X_test_eval)
y_pred_eval = model.predict(X_test_scaled_eval)
y_prob_eval = model.predict_proba(X_test_scaled_eval)[:, 1]

app = dash.Dash(__name__,
                external_scripts=[
                    'https://cdn.tailwindcss.com'
                ],
                external_stylesheets=[
                    {
                        'href': 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
                        'rel': 'stylesheet'
                    }
                ]
)
app.title = "Stock Movement Prediction Dashboard"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body class="font-sans antialiased bg-gray-100 min-h-screen p-4 md:p-8">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_card(title, content):
    return html.Div(
        className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300",
        children=[
            html.H3(title, className="text-xl font-semibold text-gray-800 mb-4"),
            content
        ]
    )

app.layout = html.Div(
    className="container mx-auto",
    children=[
        # Header
        html.H1(
            "Stock Movement Prediction Dashboard",
            className="text-4xl font-extrabold text-center text-blue-800 mb-4 mt-4"
        ),
        html.P(
            """
            This dashboard provides insights into factors influencing stocks' three day returns following earnings releases 
            and showcases a Logistic Regression model's ability to predict the direction of these returns. Explore descriptive
            statistics, model performance metrics, feature importances, and try out the interactive prediction tool.
            """,
            className="text-center text-gray-700 text-lg mb-8 max-w-3xl mx-auto"
        ),

        # Section 1: Key Descriptive Statistics
        html.H2(
            "Descriptive Statistics (Filter by Stock Movement)",
            className="text-3xl font-bold text-center text-blue-700 mb-6"
        ),
        html.Div(
            className="flex justify-center mb-6",
            children=[
                dcc.RadioItems(
                    id='movement-filter-radio',
                    options=[
                        {'label': '  All Stocks', 'value': 'all'},
                        {'label': '  Up Movement', 'value': 'up'},
                        {'label': '  Down Movement', 'value': 'down'}
                    ],
                    value='all', 
                    labelStyle={'display': 'inline-block', 'margin-right': '20px', 'font-size': '1.1em', 'color': '#374151'},
                    className="p-4 bg-white rounded-lg shadow-inner"
                )
            ]
        ),
        html.Div(
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8",
            children=[
                create_card("Surprise Scaled Distribution",
                            dcc.Graph(id='surprise-scaled-dist-plot', figure={})),
                create_card("Volatility Distribution",
                            dcc.Graph(id='volatility-dist-plot', figure={})),
                create_card("Market Return Distribution",
                            dcc.Graph(id='market-return-dist-plot', figure={})),
                create_card("Surprise Direction Counts",
                            dcc.Graph(id='surprise-direction-counts-plot', figure={})),
                create_card("Up/Down Movement Counts",
                            dcc.Graph(id='up-down-counts-plot', figure={})),
                create_card("Summary Statistics",
                            dash_table.DataTable(
                                id='summary-statistics-table',
                                columns=[{"name": i, "id": i} for i in df[SELECTED_FEATURES].describe().reset_index().columns],
                                data=[], # Data will be populated by callback
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'fontFamily': 'Inter',
                                    'fontSize': '14px',
                                    'backgroundColor': '#f9fafb',
                                    'color': '#374151'
                                },
                                style_header={
                                    'backgroundColor': '#e5e7eb',
                                    'fontWeight': 'bold',
                                    'color': '#1f2937'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': '#ffffff'
                                    }
                                ]
                            ))
            ]
        ),

        # Section 2: Model Insights
        html.H2(
            "Model Insights & Performance",
            className="text-3xl font-bold text-center text-blue-700 mb-6"
        ),
        html.Div(
            className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8",
            children=[
                create_card("Logistic Regression Coefficients",
                            dcc.Graph(id='coef-plot')),
                create_card("Confusion Matrix",
                            dcc.Graph(id='confusion-matrix-plot')),
                create_card("Classification Report",
                            dash_table.DataTable(
                                id='classification-report-table',
                                columns=[{"name": i, "id": i} for i in ['metric', 'down', 'up', 'accuracy', 'macro avg', 'weighted avg']],
                                data=[],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'fontFamily': 'Inter',
                                    'fontSize': '14px',
                                    'backgroundColor': '#f9fafb',
                                    'color': '#374151'
                                },
                                style_header={
                                    'backgroundColor': '#e5e7eb',
                                    'fontWeight': 'bold',
                                    'color': '#1f2937'
                                }
                            )),
                create_card("ROC Curve",
                            dcc.Graph(id='roc-curve-plot'))
            ]
        ),

        # Section 3: Interactive Prediction Tool
        html.H2(
            "Predict Stock Movement",
            className="text-3xl font-bold text-center text-blue-700 mb-6"
        ),
        create_card("Input Features for Prediction",
            html.Div(className="space-y-4", children=[
                html.Div(className="grid grid-cols-1 md:grid-cols-2 gap-4", children=[
                    html.Div(children=[
                        html.Label(f"{feature}:", className="block text-gray-700 text-sm font-bold mb-1"),
                        dcc.Input(
                            id=f'input-{feature.replace(" ", "-").replace("_", "-").lower()}',
                            type='number',
                            placeholder=f'Enter {feature}',
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline focus:border-blue-500",
                            value=0.0
                        )
                    ]) for feature in SELECTED_FEATURES
                ]),
                html.Button(
                    'Predict Movement',
                    id='predict-button',
                    n_clicks=0,
                    className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg shadow-lg transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 mt-4"
                ),
                html.Div(id='prediction-output', className="text-center mt-4 text-xl font-bold text-gray-900")
            ])
        )
    ]
)


# Callback to update Descriptive Statistics based on filter
@app.callback(
    [Output('surprise-scaled-dist-plot', 'figure'),
     Output('volatility-dist-plot', 'figure'),
     Output('market-return-dist-plot', 'figure'),
     Output('surprise-direction-counts-plot', 'figure'),
     Output('up-down-counts-plot', 'figure'),
     Output('summary-statistics-table', 'data')],
    [Input('movement-filter-radio', 'value')]
)
def update_descriptive_stats(selected_movement):
    if df.empty:
        return {}, {}, {}, {}, {}, []

    filtered_df = df.copy()
    plot_color = COLORS['blue'] # Default color for 'all' or when not filtered
    if selected_movement == 'up':
        filtered_df = df[df[TARGET_COLUMN] == 'up']
        plot_color = COLORS['emerald'] # Emerald for 'up'
    elif selected_movement == 'down':
        filtered_df = df[df[TARGET_COLUMN] == 'down']
        plot_color = COLORS['red'] # Red for 'down'

    # Update Histograms
    surprise_scaled_fig = px.histogram(filtered_df, x='Surprise scaled', nbins=50,
                                       title=f'Distribution of Scaled EPS Surprise ({selected_movement.capitalize()})',
                                       template='plotly_white')
    surprise_scaled_fig.update_traces(marker_color=plot_color)
    surprise_scaled_fig.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))

    volatility_fig = px.histogram(filtered_df, x='Volatility', nbins=50,
                                  title=f'Distribution of Stock Volatility ({selected_movement.capitalize()})',
                                  template='plotly_white')
    volatility_fig.update_traces(marker_color=plot_color)
    volatility_fig.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))

    market_return_fig = px.histogram(filtered_df, x='Market Return', nbins=50,
                                     title=f'Distribution of Market Return ({selected_movement.capitalize()})',
                                     template='plotly_white')
    market_return_fig.update_traces(marker_color=plot_color)
    market_return_fig.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))

    # Update Bar Charts
    surprise_direction_counts_data = pd.DataFrame({
        'Direction': filtered_df['Surprise Direction'].value_counts(normalize=True).index,
        'Proportion': filtered_df['Surprise Direction'].value_counts(normalize=True).values
    })
    surprise_direction_fig = px.bar(surprise_direction_counts_data,
                                    x='Direction', y='Proportion',
                                    labels={'Direction': 'Surprise Direction (0=Down, 1=Up)', 'Proportion': 'Proportion'},
                                    title=f'Proportion of Surprise Directions ({selected_movement.capitalize()})',
                                    template='plotly_white')
    surprise_direction_fig.update_traces(marker_color=plot_color)
    surprise_direction_fig.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))


    # Special handling for 'Up/Down Movement Counts' plot:
    # If 'all' is selected, keep distinct colors for 'up' and 'down'.
    # If 'up' or 'down' is selected, use the chosen filter color.
    up_down_counts_data = pd.DataFrame({
        'Movement': filtered_df[TARGET_COLUMN].value_counts(normalize=True).index,
        'Proportion': filtered_df[TARGET_COLUMN].value_counts(normalize=True).values
    })

    if selected_movement == 'all':
        # Define a color map specifically for 'up' and 'down' when 'all' is selected
        up_down_color_map = {
            'up': COLORS['emerald'],
            'down': COLORS['red']
        }
        up_down_fig = px.bar(up_down_counts_data,
                             x='Movement', y='Proportion',
                             labels={'Movement': 'Stock Movement', 'Proportion': 'Proportion'},
                             title=f'Proportion of Stock Movements ({selected_movement.capitalize()})',
                             color='Movement',
                             color_discrete_map=up_down_color_map,
                             template='plotly_white')
    else:
        # If filtered to 'up' or 'down', use the single selected color
        up_down_fig = px.bar(up_down_counts_data,
                             x='Movement', y='Proportion',
                             labels={'Movement': 'Stock Movement', 'Proportion': 'Proportion'},
                             title=f'Proportion of Stock Movements ({selected_movement.capitalize()})',
                             template='plotly_white')
        up_down_fig.update_traces(marker_color=plot_color)

    up_down_fig.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))


    # Update Summary Statistics Table
    summary_stats_data = filtered_df[SELECTED_FEATURES].describe().reset_index().to_dict('records')

    return (
        surprise_scaled_fig,
        volatility_fig,
        market_return_fig,
        surprise_direction_fig,
        up_down_fig,
        summary_stats_data
    )


# Callback to update Model Performance Graphs
@app.callback(
    [Output('coef-plot', 'figure'),
     Output('confusion-matrix-plot', 'figure'),
     Output('classification-report-table', 'data'),
     Output('roc-curve-plot', 'figure')],
    [Input('movement-filter-radio', 'id')]
)
def update_model_performance_graphs(_):
    if model is None or scaler is None or X_test_eval is None:
        return {}, {}, [], {}

    coefs = model.coef_[0]
    coef_df = pd.DataFrame({'Feature': SELECTED_FEATURES, 'Coefficient': coefs})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    coef_figure = px.bar(coef_df, x='Coefficient', y='Feature',
                         title='Logistic Regression Coefficients (Scaled Features)',
                         template='plotly_white')
    coef_figure.update_layout(margin=dict(t=50, b=50, l=50, r=50))

    cm = confusion_matrix(y_test_eval, y_pred_eval, labels=model.classes_)
    cm_figure = px.imshow(cm,
                          labels=dict(x="Predicted", y="True", color="Count"),
                          x=model.classes_,
                          y=model.classes_,
                          text_auto=True,
                          color_continuous_scale='Viridis',
                          title='Confusion Matrix',
                          template='plotly_white')
    cm_figure.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))
    cm_figure.update_xaxes(side="bottom")

    report = classification_report(y_test_eval, y_pred_eval, output_dict=True)
    display_report = {
        'metric': ['precision', 'recall', 'f1-score', 'support'],
        'down': [report['down']['precision'], report['down']['recall'], report['down']['f1-score'], report['down']['support']],
        'up': [report['up']['precision'], report['up']['recall'], report['up']['f1-score'], report['up']['support']],
        'accuracy': ['', '', report['accuracy'], report['accuracy']],
        'macro avg': [report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'], report['macro avg']['support']],
        'weighted avg': [report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score'], report['weighted avg']['support']]
    }
    report_data = pd.DataFrame(display_report).to_dict('records')

    fpr, tpr, thresholds = roc_curve(y_test_eval, y_prob_eval, pos_label=model.classes_[1])
    auc_score = roc_auc_score(y_test_eval, y_prob_eval)
    roc_figure = go.Figure(data=[
        go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_score:.4f})'),
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guessing (AUC = 0.5)',
                   line=dict(dash='dash', color='red'))
    ])
    roc_figure.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        margin=dict(t=50, b=50, l=50, r=50),
        template='plotly_white'
    )

    return coef_figure, cm_figure, report_data, roc_figure


@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State(f'input-{feature.replace(" ", "-").replace("_", "-").lower()}', 'value') for feature in SELECTED_FEATURES]
)
def predict_stock_movement(n_clicks, *input_values):
    if n_clicks > 0 and model is not None and scaler is not None:
        try:
            input_df = pd.DataFrame([input_values], columns=SELECTED_FEATURES)
            input_df = input_df.astype(float)
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0]
            predicted_class_prob = probability[np.where(model.classes_ == prediction)[0]][0]

            return f"Predicted Movement: {prediction.upper()} (Probability: {predicted_class_prob:.2%})"
        except Exception as e:
            return f"Error during prediction: {e}. Please check your inputs."
    return "Enter values and click 'Predict Movement' to get a prediction."


# --- 5. Run the Dash App ---
if __name__ == '__main__':
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    app.run_server(debug=True, host='0.0.0.0', port=8040)
