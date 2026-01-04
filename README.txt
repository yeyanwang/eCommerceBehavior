============================================================
README.txt — team020final
Fall 2025 CSE6242 Term Project
============================================================

====================
DESCRIPTION
====================
This package contains the full implementation of the interactive 
E-Commerce Behavior Analytics Dashboard. The system integrates 
descriptive analytics, predictive modeling, and clustering-based user 
segmentation into a unified Dash web application. The dashboard enables 
users to explore large-scale, session-level e-commerce data through 
visual summaries, machine-learning insights, and dynamic behavioral 
segmentation.

The application includes three main components:

1. Descriptive Analytics
   - KPI cards summarizing key behavioral metrics.
   - Time-series visualizations (daily, hourly, weekday patterns).
   - Funnel view of user progression from viewing to carting to purchasing.
   - Interactive filters for category, brand, and price range.
   - Dynamic charts that update based on user-selected filters.

2. Predictive Analytics
   - Heatmap of predicted conversion probability by hour and day of week.
   - Logistic regression feature-importance visualization showing 
     coefficients, odds ratios, and intuitive explanations.

3. Clustering Analytics
   - 3D PCA scatter plot of session-level behavioral embeddings.
   - Cluster-conditioned boxplots for purchase frequency, viewing behavior, 
     recency, and average price viewed.

NOTE: All visual layers load directly from preprocessed Parquet files created in Final_Analysis_Notebook.ipynb. These files store feature engineering, model outputs, and aggregates from the public 2019 October e-commerce sessions dataset. Parquet formatting ensures fast load times and keeps the dashboard responsive.

====================
INSTALLATION
====================
1. Install Python 3.9+ in your local Anaconda environment (recommended)
   or any preferred Python 3 environment.

2. With the same environment activated, install the required packages:

   pip install dash==2.14.2
   pip install plotly
   pip install pandas
   pip install numpy
   pip install fastparquet

====================
EXECUTION
====================
1. Ensure all required Parquet files are in the same directory as app.py - these can be found in the CODE folder of team020final :

   daily_events.parquet
   hourly_events.parquet
   dow_events.parquet
   price_summary.parquet
   funnel_summary.parquet
   top_brands.parquet
   top_categories.parquet
   session_histogram.parquet
   session_stats_sample.parquet
   price_sample.parquet
   heatmap_data.parquet
   feature_importance_data.parquet
   pca.parquet
   cluster_features.parquet

2. Ensure an "assets" folder containing your style.css file is also in the 
   same directory as app.py.

3. In your Anaconda (or other Python) environment, open a terminal and 
   navigate to the directory containing app.py and all required files. 

4. Launch the dashboard by running:

   python app.py

Once the server starts, open a browser and navigate to:
   http://127.0.0.1:8050

Allow a few seconds for each tab to load once you open up the app in your browser.
We recommend using Chrome for best performance.

============================================================
END OF README.txt
============================================================
