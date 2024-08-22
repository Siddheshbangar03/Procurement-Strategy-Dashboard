import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet  # Import Prophet
from prophet.plot import plot_plotly

# Load Data
@st.cache_data
def load_data():
    supplier_df = pd.read_csv('walmart_supplier_order_data.csv')
    sell_df = pd.read_csv('walmart_product_sell_data.csv')
    return supplier_df, sell_df

supplier_df, sell_df = load_data()

# Supplier Performance Analysis
def compute_supplier_performance(supplier_df, sell_df):
    if 'Product Name' not in sell_df.columns:
        st.error("The 'Product Name' column is missing in the sell_df.")
        return pd.DataFrame()
    
    # Compute average sale price for each product
    average_sale_price = sell_df.groupby('Product Name')['Sale Price'].mean().reset_index()
    average_sale_price.columns = ['Product Name', 'Average Sale Price']
    
    # Merge the average sale price with supplier data
    supplier_df = supplier_df.merge(average_sale_price, on='Product Name', how='left')
    
    # Calculate Cost Efficiency (Used only for calculations, not displayed)
    supplier_df['Cost Efficiency'] = supplier_df['Purchase Price'] / supplier_df['Average Sale Price']
    
    # Compute reliability score for each supplier
    reliability = supplier_df.groupby('Supplier Name')['OnTime Delivery'].mean().reset_index()
    reliability.columns = ['Supplier Name', 'Reliability']
    
    # Merge reliability with supplier data
    supplier_df = supplier_df.merge(reliability, on='Supplier Name', how='left')
    
    # Aggregate quantity sold by product and supplier
    product_performance = sell_df[['Product Name', 'Supplier Name', 'Quantity Sold', 'Date']].copy()
    product_performance['Date'] = pd.to_datetime(product_performance['Date'])
    product_performance['Month'] = product_performance['Date'].dt.to_period('M')
    product_performance = product_performance.groupby(['Product Name', 'Supplier Name', 'Month']).agg({
        'Quantity Sold': 'sum'
    }).reset_index()
    
    # Convert Period to Timestamp for Prophet compatibility
    product_performance['Month'] = product_performance['Month'].apply(lambda x: x.to_timestamp())

    # Merge product performance with supplier data
    supplier_df = supplier_df.merge(product_performance, on=['Product Name', 'Supplier Name'], how='left')
    
    # Remove any duplicate rows based on Supplier Name, Product Name
    supplier_df = supplier_df.drop_duplicates(subset=['Supplier Name', 'Product Name'])
    
    # Compute performance score
    weights = {'Cost Efficiency': 0.4, 'Reliability': 0.6}
    supplier_df['Performance Score'] = (weights['Cost Efficiency'] * supplier_df['Cost Efficiency'].fillna(0)) + \
                                        (weights['Reliability'] * supplier_df['Reliability'].fillna(0))
    
    return supplier_df, product_performance

performance_df, product_performance = compute_supplier_performance(supplier_df, sell_df)

st.title("Walmart Procurement Strategy Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Supplier Performance Analysis", "Procurement Details"])

# Supplier Performance Analysis
if page == "Supplier Performance Analysis":
    st.header("Supplier Performance Analysis")

    # Select Product
    products = performance_df['Product Name'].unique()
    selected_product = st.selectbox('Select Product', options=['All'] + list(products))

    if selected_product != 'All':
        # Filter Data
        filtered_df = performance_df[performance_df['Product Name'] == selected_product]

        # Replace 1 with Yes and 0 with No in OnTime Delivery
        filtered_df['OnTime Delivery'] = filtered_df['OnTime Delivery'].apply(lambda x: 'Yes' if x == 1 else 'No')

        # Remove Product Name and Cost Efficiency columns from the display
        filtered_df_display = filtered_df.drop(columns=['Product Name', 'Cost Efficiency'])

        # Display Filtered Data
        st.subheader("Supplier Performance Data")
        st.dataframe(filtered_df_display[['Supplier Name', 'Reliability', 'Quantity Sold', 'Performance Score', 'OnTime Delivery']], width=1000)

        # Show the best supplier based on Performance Score, excluding those with 'On Time Delivery' as 'No'
        filtered_df_best_supplier = filtered_df[filtered_df['OnTime Delivery'] == 'Yes']
        if not filtered_df_best_supplier.empty:
            best_supplier = filtered_df_best_supplier.loc[filtered_df_best_supplier['Performance Score'].idxmax()]
            st.subheader("Best Supplier for Selected Product")
            st.write(f"Supplier Name: {best_supplier['Supplier Name']}")
            st.write(f"Performance Score: {best_supplier['Performance Score']}")
        else:
            st.write("No suppliers with 'On Time Delivery' status available for the selected product.")

        # Visualization of Supplier Performance
        st.subheader("Supplier Performance Visualization")

        # Bar plot for Performance Score
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Supplier Name', y='Performance Score', data=filtered_df_display, palette='coolwarm')
        plt.xticks(rotation=45)
        plt.title('Supplier Performance Score for Selected Product')
        plt.xlabel('Supplier Name')
        plt.ylabel('Performance Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(plt.gcf())

        # Bar plot for Quantity Sold
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Supplier Name', y='Quantity Sold', data=filtered_df_display, palette='coolwarm')
        plt.xticks(rotation=45)
        plt.title('Quantity Sold by Supplier for Selected Product')
        plt.xlabel('Supplier Name')
        plt.ylabel('Quantity Sold')
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(plt.gcf())

        # Bar plot for Reliability
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Supplier Name', y='Reliability', data=filtered_df_display, palette='coolwarm')
        plt.xticks(rotation=45)
        plt.title('Supplier Reliability for Selected Product')
        plt.xlabel('Supplier Name')
        plt.ylabel('Reliability')
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(plt.gcf())

        # Pie chart for On Time Delivery
        st.subheader("On Time Delivery Pie Chart")
        delivery_counts = filtered_df_display['OnTime Delivery'].value_counts()
        plt.figure(figsize=(2, 2))
        plt.pie(delivery_counts, labels=delivery_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
        st.pyplot(plt.gcf())

    else:
        st.write("Please select a product to view the details.")


# Procurement Details Section
elif page == "Procurement Details":
    st.header("Product-Wise Procurement Details")

    # Category and Product Name Selectors
    categories = sell_df['Category'].unique()
    selected_category = st.selectbox('Select Category', options=['All'] + list(categories))
    filtered_sell_df = sell_df if selected_category == 'All' else sell_df[sell_df['Category'] == selected_category]

    products = filtered_sell_df['Product Name'].unique()
    selected_product = st.selectbox('Select Product', options=['All'] + list(products))

    if selected_product != 'All':
        filtered_sell_df = filtered_sell_df[filtered_sell_df['Product Name'] == selected_product]

        # Check and remove specific columns if they exist
        columns_to_drop = ['Product Name', 'Product Url', 'Brand', 'Category']
        columns_present = [col for col in columns_to_drop if col in filtered_sell_df.columns]
        filtered_sell_df_display = filtered_sell_df.drop(columns=columns_present)

        st.subheader("Product Procurement Details")
        st.dataframe(filtered_sell_df_display, width=1000)

        # Month Counter for Procurement
        month_counter = st.slider('Select Number of Months for Forecast', min_value=1, max_value=12, value=3)

        # Prepare data for Prophet
        product_data = filtered_sell_df[['Date', 'Quantity Sold']].copy()
        product_data['Date'] = pd.to_datetime(product_data['Date'])
        product_data['Month'] = product_data['Date'].dt.to_period('M')
        product_data = product_data.groupby('Month').agg({'Quantity Sold': 'sum'}).reset_index()
        product_data.columns = ['ds', 'y']
        
        # Convert Period to Timestamp for Prophet
        product_data['ds'] = product_data['ds'].apply(lambda x: x.to_timestamp())

        # Prophet Model
        model = Prophet()
        model.fit(product_data)

        # Future DataFrame
        future_df = model.make_future_dataframe(periods=month_counter, freq='M')

        # Forecast
        forecast = model.predict(future_df)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(month_counter)
        forecast_df.columns = ['Date', 'Forecast Quantity', 'Lower Bound', 'Upper Bound']
        
        # Ensure non-negative quantities and round to whole numbers
        forecast_df['Forecast Quantity'] = forecast_df['Forecast Quantity'].clip(lower=0).round()
        forecast_df['Lower Bound'] = forecast_df['Lower Bound'].clip(lower=0).round()
        forecast_df['Upper Bound'] = forecast_df['Upper Bound'].clip(lower=0).round()

        st.subheader("Procurement Forecast Table")
        st.dataframe(forecast_df, width=1000)

        # Historical Quantity Sold Visualization
        st.subheader("Historical Quantity Sold with Forecast")
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(product_data['ds'], product_data['y'], label='Historical Quantity Sold', color='#2ca02c', marker='o')
        
        # Plot forecast data
        plt.plot(forecast_df['Date'], forecast_df['Forecast Quantity'], label='Forecast Quantity', color='#1f77b4', marker='o')
        plt.fill_between(forecast_df['Date'], forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='#1f77b4', alpha=0.3)

        plt.title('Historical Quantity Sold and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Quantity Sold')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

        # Best Supplier Summary
        best_supplier = performance_df[(performance_df['Product Name'] == selected_product) & (performance_df['OnTime Delivery'] == 1)].sort_values(by='Performance Score', ascending=False).head(1)
        if not best_supplier.empty:
            st.subheader("Best Supplier Summary")
            best_supplier = best_supplier.iloc[0]
            st.write(f"Best Supplier: {best_supplier['Supplier Name']}")
            st.write(f"Best Price: ${best_supplier['Purchase Price']:.2f}")
            st.write(f"On Time Delivery: {'Yes' if best_supplier['OnTime Delivery'] == 1 else 'No'}")
            st.write(f"Performance Score: {best_supplier['Performance Score']:.2f}")
        else:
            st.write("No supplier data available for the selected product.")
        
    else:
        st.write("Please select a product to view the details.")
