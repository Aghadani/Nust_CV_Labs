"""
Assessment Question 1: Stock Price Prediction
==============================================

INSTRUCTIONS FOR STUDENTS:
This code contains MULTIPLE errors that prevent it from running correctly.
Your task is to debug and fix ALL errors to make the program work properly.

**DEBUGGING CHECKLIST:**
□ Check all variable names (spelling and consistency)
□ Check function parameters and arguments
□ Check data types (numbers vs strings)
□ Check mathematical operations
□ Check array/list indexing
□ Look for typos in method names
□ Verify all imports are used correctly

Student Name: ____________________
CMSID: ____________________
Date: ____________________

GRADING: 
- Finding and fixing errors: 10 points
- Code runs without errors: Bonus points
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# PART A: DATA PREPARATION (2 marks)
# ==============================================================================

def create_stock_dataset(num_samples=50, save_csv=True):
    """
    Create a realistic stock price dataset.
    """
    print("=" * 60)
    print("PART A: Data Preparation")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base price trend
    days = np.arange(num_samples)
    base_price = 100 + days * 0.5
    
    # Add some volatility
    volatility = np.random.randn(num_samples) * 2
    
    # Generate Open prices
    open_prices = base_price + volatility
    
    # Generate High prices (slightly higher than open)
    high_offset = np.abs(np.random.randn(num_samples)) * 3
    high_prices = open_prices + high_offset
    
    # Generate Low prices (slightly lower than open)
    low_offset = np.abs(np.random.randn(num_samples)) * 3
    low_prices = open_price - low_offset  # ERROR 1: Variable name typo
    
    # Generate Close prices (between low and high, closer to open)
    close_prices = (
        0.5 * open_prices + 
        0.25 * high_prices + 
        0.25 * low_prices +
        np.random.randn(num_samples) * 1
    )
    
    # Ensure Close is between Low and High
    close_prices = np.clip(close_prices, low_prices, high_prices)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    })
    
    # Round to 2 decimal places
    data = data.round('2')  # ERROR 2: Should be integer not string
    
    print(f"\nDataset created with {num_samples} samples")
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nDataset statistics:")
    print(data.describe())
    
    # Save to CSV
    if save_csv:
        filename = 'stock_data.csv'
        data.to_csv(filename, index=False)
        print(f"\n✓ Dataset saved as '{filename}'")
    
    return data

# ==============================================================================
# PART B: MODEL IMPLEMENTATION (4 marks)
# ==============================================================================

def train_stock_model(data):
    """
    Train a Multiple Linear Regression model on stock data.
    """
    print("\n" + "=" * 60)
    print("PART B: Model Implementation")
    print("=" * 60)
    
    # Prepare features (X) and target (y)
    X = data[['Open', 'High', 'Low']].values
    y = data['Close'].values
    
    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    # Split into training and testing sets (70-30 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state="42"  # ERROR 3: Should be int not string
    )
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    # Create and train the model
    print("\nTraining Multiple Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✓ Model training completed!")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    print(f"\nPredictions made on {len(y_pred)} test samples")
    
    return model, X_train, X_test, y_train, y_test, y_pred

# ==============================================================================
# PART C: MODEL EVALUATION (2 marks)
# ==============================================================================

def evaluate_model(model, y_test, y_pred):
    """
    Evaluate the model and display metrics.
    """
    print("\n" + "=" * 60)
    print("PART C: Model Evaluation")
    print("=" * 60)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, ypred)  # ERROR 4: Variable name typo
    
    # Display metrics
    print("\n" + "-" * 60)
    print("EVALUATION METRICS")
    print("-" * 60)
    print(f"Mean Absolute Error (MAE):        ${mae:.4f}")
    print(f"Mean Squared Error (MSE):         ${mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):   ${rmse:.4f}")
    print(f"R-squared (R²) Score:             {r2:.4f}")
    print("-" * 60)
    
    # Interpret R² score
    print("\nModel Performance Interpretation:")
    if r2 > 0.9:
        print(f"  Excellent! R² = {r2:.4f} indicates the model explains {r2*100:.2f}% of variance")
    elif r2 > 0.7:
        print(f"  Good! R² = {r2:.4f} indicates the model explains {r2*100:.2f}% of variance")
    elif r2 > 0.5:
        print(f"  Moderate. R² = {r2:.4f} indicates the model explains {r2*100:.2f}% of variance")
    else:
        print(f"  Weak. R² = {r2:.4f} indicates the model explains only {r2*100:.2f}% of variance")
    
    # Display model coefficients
    print("\n" + "-" * 60)
    print("MODEL COEFFICIENTS")
    print("-" * 60)
    print(f"Intercept (b):                    ${model.intercept_:.4f}")
    print(f"\nFeature Weights:")
    feature_names = ['Open Price', 'High Price', 'Low Price']
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name:<20} {coef:>10.4f}")
    print("-" * 60)
    
    # Equation
    print("\nRegression Equation:")
    print(f"Close = {model.intercept_:.4f} + "
          f"{model.coef_[0]:.4f}*Open + "
          f"{model.coef_[1]:.4f}*High + "
          f"{model.coef_[2]:.4f}*Low")
    
    return mae, mse, rmse, r2

# ==============================================================================
# PART D: VISUALIZATION (2 marks)
# ==============================================================================

def create_visualizations(y_test, y_pred):
    """
    Create visualization plots for the results.
    """
    print("\n" + "=" * 60)
    print("PART D: Visualization")
    print("=" * 60)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # -------------------------------------------------------------------------
    # Plot 1: Actual vs Predicted
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    # Scatter plot
    ax1.scatter(y_test, y_pred, alpha=0.6, s=80, edgecolors='black', linewidth=1)
    
    # Perfect prediction line (y=x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and title
    ax1.set_xlabel('Actual Close Price ($)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Close Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Actual vs Predicted Stock Prices', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add R² annotation
    r2 = r2_score(y_test, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=ax1.transAxes,
             fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # -------------------------------------------------------------------------
    # Plot 2: Residual Plot
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    # Calculate residuals (errors)
    residuals = y_test - y_pred
    
    # Scatter plot of residuals
    ax2.scatter(y_pred, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=1)
    
    # Zero line
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    
    # Labels and title
    ax2.set_xlabel('Predicted Close Price ($)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Actual - Predicted) ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot (Prediction Errors)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    stats_text = f'Mean: ${mean_residual:.2f}\nStd Dev: ${std_residual:.2f}'
    ax2.text(0.05, 0.95, stats_text,
             transform=ax2.transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined figure
    filename1 = 'actual_vs_predicted.png'
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {filename1}")
    
    plt.show()
    
    # -------------------------------------------------------------------------
    # Create separate residual plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_pred, residuals, alpha=0.6, s=80, 
                color='steelblue', edgecolors='black', linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Predicted Close Price ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (Actual - Predicted) ($)', fontsize=12, fontweight='bold')
    plt.title('Residual Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f'Mean Residual: ${mean_residual:.4f}\n'
    stats_text += f'Std Deviation: ${std_residual:.4f}\n'
    stats_text += f'Max Error: ${np.max(np.abs(residuals)):.4f}'
    
    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             fontsize=11,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    filename2 = 'residual_plot.png'
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    
    plt.show()
    
    print("\n✓ All visualizations created successfully!")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main function to execute the complete solution.
    """
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "STOCK PRICE PREDICTION - LINEAR REGRESSION" + " " * 9 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Part A: Create dataset
    dataset = create_stock_dataset(num_samples=50, save_csv=True)  # ERROR 5: Wrong variable name
    
    # Part B: Train model
    model, X_train, X_test, y_train, y_test, y_pred = train_stock_model(data)  # ERROR 6: 'data' not defined
    
    # Part C: Evaluate model
    mae, mse, rmse, r2 = evaluate_model(model, y_test, y_pred)
    
    # Part D: Create visualizations
    create_visualizations(y_test, y_pred)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SOLUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  1. stock_data.csv - Stock price dataset")
    print("  2. actual_vs_predicted.png - Comparison plot")
    print("  3. residual_plot.png - Error analysis")
    print()
    print("Key Results:")
    print(f"  - R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"  - RMSE: ${rmse:.4f}")
    print(f"  - MAE: ${mae:.4f}")
    print()
    print("Interpretation:")
    print("  The model successfully predicts stock closing prices")
    print("  based on open, high, and low prices with good accuracy.")
    print()

if __name__ == "__main__":
    main()
