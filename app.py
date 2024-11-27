import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def main():
    st.title("Multi-Correlate Data Analysis")
    st.write("Upload multiple files to combine them for analysis.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your files (CSV/Excel)", 
        type=["csv", "xlsx"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        dataframes = []
        all_columns = set()

        # Process each file
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file, parse_dates=True, infer_datetime_format=True)
                elif file.name.endswith('.xlsx'):
                    df = pd.read_excel(file, parse_dates=True, infer_datetime_format=True)

                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                all_columns.update(df.columns)
                dataframes.append(df)

                st.write(f"Preview of {file.name}:")
                st.write(df.head())

            except Exception as e:
                st.error(f"Error reading file {file.name}: {e}")

        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on='date', how='outer')

        st.write("Merged Dataframe:")
        st.write(merged_df.head())

        # Drop 'date' from numeric columns
        numeric_columns = merged_df.select_dtypes(include=['number']).columns.tolist()

        # Allow user to select variables for correlation
        selected_columns = st.multiselect(
            "Select variables for correlation analysis",
            options=numeric_columns,
            default=numeric_columns
        )

        if st.button("Show Correlation Matrix"):
            if len(selected_columns) < 2:
                st.warning("Please select at least two variables for correlation analysis.")
            else:
                st.write("Correlation Matrix:")

                correlation_matrix = merged_df[selected_columns].corr()
                st.write(correlation_matrix)

                # Find pairs with significant correlation (absolute value >= 0.5)
                significant_pairs = [
                    (col1, col2, correlation_matrix.loc[col1, col2])
                    for i, col1 in enumerate(correlation_matrix.columns)
                    for j, col2 in enumerate(correlation_matrix.columns)
                    if i < j and abs(correlation_matrix.loc[col1, col2]) >= 0.5
                ]

                # Display significant pairs
                if significant_pairs:
                    st.subheader("Variable Pair Significant Correlation (|correlation| >= 0.5)")
                    st.table(pd.DataFrame(significant_pairs, columns=["Variable 1", "Variable 2", "Correlation"]))
                else:
                    st.write("No significant correlations found.")

                # Dynamic text color function
                def text_color(value):
                    return 'black' if value > 0.5 else 'white'

                # Create a 2D list of text colors based on correlation values
                text_colors = np.vectorize(text_color)(correlation_matrix.values)

                fig = go.Figure(
                    data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        colorscale="Viridis",
                        colorbar=dict(title="Correlation"),
                        text=correlation_matrix.round(2).values,  # Add rounded correlation values
                        hovertemplate='%{x} - %{y}: %{text}',  # Show values on hover
                        showscale=True
                    )
                )

                # Dynamically set the color of each cell's text
                for i in range(len(correlation_matrix.columns)):
                    for j in range(len(correlation_matrix.index)):
                        fig.add_trace(go.Scatter(
                            x=[correlation_matrix.columns[i]],
                            y=[correlation_matrix.index[j]],
                            text=[f"{correlation_matrix.iloc[i, j]:.2f}"],  # Round to 2 decimals
                            mode="text",
                            textfont=dict(color=text_colors[i][j], size=12),
                            showlegend=False
                        ))

                fig.update_layout(
                    title="Correlation Matrix Heatmap",
                    xaxis=dict(title="Features"),
                    yaxis=dict(title="Features"),
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
