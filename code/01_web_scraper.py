"""
GitHub Flower Image Scraper
Scrapes flower images from a GitHub repository using GitHub API
"""

import os
import requests
import time

class GitHubFlowerScraper:
    def __init__(self, base_dir='data/flowers'):
        self.base_dir = base_dir
        self.flower_types = ['sunflower', 'daisy', 'dandelion', 'rose', 'tulip']
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def create_folder_structure(self):
        """Create necessary folders for each flower type"""
        print("\n📁 Creating folder structure...")
        for flower in self.flower_types:
            folder_path = os.path.join(self.base_dir, flower)
            os.makedirs(folder_path, exist_ok=True)
            print(f"  ✓ Created: {folder_path}")
    
    def parse_github_url(self, github_url):
        """
        Parse GitHub URL and return API endpoint
        Input: https://github.com/username/repo
        Output: API URL for contents
        """
        github_url = github_url.rstrip('/')
        
        # Remove https://github.com/
        path = github_url.replace('https://github.com/', '')
        parts = path.split('/')
        
        if len(parts) >= 2:
            username = parts[0]
            repo = parts[1]
            
            # API base URL
            api_url = f"https://api.github.com/repos/{username}/{repo}/contents"
            
            print(f"\n🔍 Parsed GitHub Repository:")
            print(f"  Username: {username}")
            print(f"  Repository: {repo}")
            print(f"  API URL: {api_url}")
            
            return api_url, username, repo
        
        return None, None, None
    
    def get_folder_contents(self, api_url):
        """Get list of folders/files from GitHub API"""
        try:
            response = requests.get(api_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                print(f"  ⚠ Rate limit exceeded. Wait 60 seconds...")
                time.sleep(60)
                return self.get_folder_contents(api_url)
            else:
                print(f"  ✗ Error: HTTP {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return None
        
        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
            return None
    
    def download_image(self, download_url, flower_name, original_filename):
        """Download a single image from GitHub raw URL"""
        try:
            response = requests.get(download_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                # Create filename with timestamp to avoid duplicates
                timestamp = int(time.time() * 1000)
                name, ext = os.path.splitext(original_filename)
                new_filename = f"{flower_name}-{timestamp}-{name}{ext}"
                
                filepath = os.path.join(self.base_dir, flower_name, new_filename)
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"    ✓ Saved: {new_filename}")
                return True
            else:
                print(f"    ✗ Failed: HTTP {response.status_code}")
                return False
        
        except Exception as e:
            print(f"    ✗ Error downloading {original_filename}: {str(e)}")
            return False
    
    def scrape_flower_folder(self, folder_url, flower_name):
        """Download all images from a specific flower folder"""
        print(f"\n  🌸 Scraping {flower_name} folder...")
        
        contents = self.get_folder_contents(folder_url)
        
        if not contents:
            print(f"    ✗ Could not access folder")
            return 0
        
        downloaded = 0
        
        for item in contents:
            # Check if it's a file (not a folder)
            if item['type'] == 'file':
                filename = item['name']
                
                # Check if it's an image file
                if any(filename.lower().endswith(ext) 
                      for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
                    
                    download_url = item['download_url']
                    
                    if self.download_image(download_url, flower_name, filename):
                        downloaded += 1
                        time.sleep(0.5)  # Be nice to GitHub servers
        
        print(f"  ✓ Downloaded {downloaded} {flower_name} images")
        return downloaded
    
    def scrape_repository(self, github_url):
        """Main scraping function"""
        print("\n" + "=" * 60)
        print("🌺 GITHUB FLOWER IMAGE SCRAPER")
        print("=" * 60)
        
        # Parse GitHub URL
        api_url, username, repo = self.parse_github_url(github_url)
        
        if not api_url:
            print("✗ Invalid GitHub URL")
            return 0
        
        # Create folder structure
        self.create_folder_structure()
        
        # Get repository contents
        print(f"\n🔍 Fetching repository contents...")
        contents = self.get_folder_contents(api_url)
        
        if not contents:
            print("✗ Could not access repository")
            print("\n💡 Make sure:")
            print("  1. Repository is PUBLIC")
            print("  2. URL is correct: https://github.com/username/repo")
            print("  3. Repository contains flower folders")
            return 0
        
        # Find flower folders
        print(f"\n📂 Looking for flower folders...")
        total_downloaded = 0
        found_folders = []
        
        for item in contents:
            if item['type'] == 'dir' and item['name'] in self.flower_types:
                found_folders.append(item['name'])
                print(f"  ✓ Found: {item['name']} folder")
                
                # Scrape this folder
                folder_url = item['url']
                downloaded = self.scrape_flower_folder(folder_url, item['name'])
                total_downloaded += downloaded
                
                time.sleep(1)  # Pause between folders
        
        # Report results
        print("\n" + "=" * 60)
        print("✅ SCRAPING COMPLETE!")
        print("=" * 60)
        print(f"Total folders found: {len(found_folders)}")
        print(f"Total images downloaded: {total_downloaded}")
        
        if len(found_folders) < len(self.flower_types):
            missing = set(self.flower_types) - set(found_folders)
            print(f"\n⚠ Missing folders: {', '.join(missing)}")
            print("   Make sure all flower folders exist in the repository")
        
        return total_downloaded
    
    def generate_scraping_log(self, github_url):
        """Generate documentation log"""
        log_path = os.path.join(self.base_dir, 'scraping_log.txt')
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("FLOWER IMAGE WEB SCRAPING LOG\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Scraping Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: GitHub Repository\n")
            f.write(f"Repository URL: {github_url}\n")
            f.write(f"Method: GitHub REST API v3\n\n")
            
            f.write("Flower Types:\n")
            f.write("-" * 60 + "\n")
            
            total_images = 0
            for flower in self.flower_types:
                folder_path = os.path.join(self.base_dir, flower)
                if os.path.exists(folder_path):
                    count = len([f for f in os.listdir(folder_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
                    f.write(f"{flower.capitalize()}: {count} images\n")
                    total_images += count
                else:
                    f.write(f"{flower.capitalize()}: 0 images (folder not found)\n")
            
            f.write(f"\nTotal Images: {total_images}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Technical Details:\n")
            f.write("- API: GitHub REST API v3\n")
            f.write("- Authentication: None (public repository)\n")
            f.write("- Rate Limit: 60 requests/hour (unauthenticated)\n")
            f.write("- Download Method: Direct raw content URLs\n\n")
            
            f.write("Legal & Ethical:\n")
            f.write("- All images from public GitHub repository\n")
            f.write("- Used for educational purposes only\n")
            f.write("- Part of AI/ML course project\n")
            f.write("- Proper attribution to original sources\n")
        
        print(f"\n✓ Scraping log saved: {log_path}")


def main():
    """Main execution"""
    scraper = GitHubFlowerScraper(base_dir='AI-CPS/data/flowers')
    
    print("\n" + "=" * 60)
    print("  GITHUB FLOWER IMAGE SCRAPER - SETUP")
    print("=" * 60)
    
    print("\n📝 INSTRUCTIONS:")
    print("1. Create a PUBLIC GitHub repository")
    print("2. Upload flower images in folders: sunflower, daisy, dandelion, rose, tulip")
    print("3. Get the repository URL")
    print("\nExample URL: https://github.com/yourusername/flower-images-dataset")
    
    print("\n" + "=" * 60)
    
    # Get GitHub URL from user
    github_url = input("\nEnter your GitHub repository URL: ").strip()
    
    if not github_url:
        print("✗ No URL provided. Exiting...")
        return
    
    # Validate URL format
    if not github_url.startswith('https://github.com/'):
        print("✗ Invalid URL format. Must start with: https://github.com/")
        return
    
    # Start scraping
    total = scraper.scrape_repository(github_url)
    
    # Generate documentation
    if total > 0:
        scraper.generate_scraping_log(github_url)
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS!")
        print("=" * 60)
        print(f"✓ Downloaded {total} images")
        print(f"✓ Created scraping log")
        print("\n📌 NEXT STEPS:")
        print("1. Check data/flowers/ folder for downloaded images")
        print("2. Review scraping_log.txt")
        print("3. Run data preparation script to create CSV files")
    else:
        print("\n✗ No images downloaded. Please check:")
        print("  - Repository is PUBLIC")
        print("  - Folders (sunflower, daisy, etc.) exist")
        print("  - Images are in the folders")


if __name__ == "__main__":
    main()