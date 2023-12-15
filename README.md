Install Required Packages:
Ensure you have the necessary Python packages installed. You can install them using the following command:
pip install lightkurve numpy

Data Download:
Go to MAST.
Navigate to MAST catalogs -> TESS CTL v8.01 -> Advanced Search.
Refine your search based on criteria like temperature and distance.
Obtain the star's ID (e.g., 140206488) and cross-check on EXOMAST for existing planets.
Update the search_targetpixelfile function with your star's ID and other parameters.
Run the code to download the TESS target pixel file.



Downloader Helper:
The downloader helper script allows you to fetch additional data from MAST for a specific product group ID. To use this helper:

Update the product_group_id variable with the desired product group ID.
Set the url variable with the appropriate MAST API link.
Specify the destination folder where you want to save the downloaded data.
Run the code to download and unzip the data.
