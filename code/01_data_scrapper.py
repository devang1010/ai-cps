"""
Data Scraper for German Credit Risk Dataset
Author: [Your Name]
Date: January 2026
Course: M. Grum: Advanced AI-based Application Systems

This script scrapes the German Credit Risk dataset from GitHub repository
using BeautifulSoup4 and saves it locally.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from io import StringIO

def scrape_github_csv(url):
    """
    Scrape CSV data from GitHub repository
    
    Args:
        url (str): GitHub URL of the CSV file
    
    Returns:
        pd.DataFrame: Scraped data as pandas DataFrame
    """
    print("🔍 Starting data scraping process...")
    print(f"📍 Target URL: {url}")
    
    try:
        # Convert GitHub blob URL to raw URL for direct access
        raw_url = url.replace('/blob/', '/raw/')
        print(f"📍 Raw URL: {raw_url}")
        
        # Send GET request with headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(raw_url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        
        print(f"✅ Successfully fetched data (Status Code: {response.status_code})")
        
        # Parse CSV content using pandas
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"❌ Error parsing CSV: {e}")
        return None


def scrape_with_beautifulsoup(url):
    """
    Alternative method: Scrape using BeautifulSoup to parse HTML table
    (Use this if direct CSV download doesn't work)
    
    Args:
        url (str): GitHub URL of the CSV file
    
    Returns:
        pd.DataFrame: Scraped data as pandas DataFrame
    """
    print("\n🔍 Using BeautifulSoup method...")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the CSV content in GitHub's HTML structure
        # GitHub renders CSV as a table with class 'csv-data'
        csv_table = soup.find('table', {'class': 'csv-data'})
        
        if csv_table:
            # Extract data from table
            data = []
            headers_row = csv_table.find('thead')
            body_rows = csv_table.find('tbody')
            
            # Get column headers
            if headers_row:
                headers = [th.get_text(strip=True) for th in headers_row.find_all('th')]
            
            # Get data rows
            if body_rows:
                for row in body_rows.find_all('tr'):
                    cols = [td.get_text(strip=True) for td in row.find_all('td')]
                    data.append(cols)
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            print(f"✅ Successfully scraped data using BeautifulSoup")
            print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return df
        else:
            print("❌ Could not find CSV table in HTML")
            return None
            
    except Exception as e:
        print(f"❌ Error with BeautifulSoup scraping: {e}")
        return None


def save_raw_data(df, output_dir='../data'):
    """
    Save scraped data to local filesystem
    
    Args:
        df (pd.DataFrame): Data to save
        output_dir (str): Directory to save the data
    
    Returns:
        str: Path to saved file
    """
    if df is None:
        print("❌ No data to save")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'german_credit_raw.csv')
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Raw data saved successfully!")
        print(f"📂 Location: {output_path}")
        print(f"📊 Saved {df.shape[0]} rows and {df.shape[1]} columns")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return None


def display_data_info(df):
    """
    Display information about the scraped dataset
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    if df is None:
        return
    
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    
    print(f"\n🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Column Names and Types:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  • {col:<20} | {str(dtype):<10} | Non-null: {non_null}/{len(df)}")
    
    print("\n🔍 First 5 rows of the dataset:")
    print(df.head())
    
    print("\n📈 Basic Statistics:")
    print(df.describe())
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  ✅ No missing values found!")


def main():
    """
    Main function to execute the scraping process
    """
    print("\n" + "="*60)
    print("🚀 GERMAN CREDIT RISK DATA SCRAPER")
    print("="*60 + "\n")
    
    # GitHub URL of the dataset
    github_url = "https://github.com/devang1010/German-credit-score/blob/main/german_credit_data.csv"
    
    # Method 1: Direct CSV download (preferred)
    df = scrape_github_csv(github_url)
    
    # Method 2: If Method 1 fails, try BeautifulSoup
    if df is None:
        print("\n⚠️  Direct method failed. Trying BeautifulSoup method...")
        df = scrape_with_beautifulsoup(github_url)
    
    # Display dataset information
    if df is not None:
        display_data_info(df)
        
        # Save the raw data
        save_raw_data(df)
        
        print("\n✅ Scraping completed successfully!")
    else:
        print("\n❌ Scraping failed. Please check the URL and try again.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()