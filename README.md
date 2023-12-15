Install Required Packages:
Ensure you have the necessary Python packages installed. You can install them using the following command:
pip install lightkurve numpy

Data Download:
Go to MAST.
![image](https://github.com/ShreyasTembhare/Exoplanet-detector/assets/77243055/cd8abf3e-c266-45d7-a5f3-4ba4946e0e16)

Navigate to MAST catalogs -> TESS CTL v8.01 -> Advanced Search.
Refine your search based on criteria like temperature and distance.
Obtain the star's ID (e.g., 140206488) and cross-check on EXOMAST for existing planets.
Update the search_targetpixelfile function with your star's ID and other parameters.
Run the code to download the TESS target pixel file.


Exploring the Data:

Utilize the plot and to_lightcurve functions to visualize a snapshot and the lightcurve.
Flatten the lightcurve using the flatten method for better analysis.
Periodogram Analysis:

Use the Box Least Squares (BLS) algorithm to find the period of potential orbiting objects.
Adjust the period array in the code to refine the search.
Identify the period, transit time, and duration of the most prominent orbiting object.
Phase-Fold the Lightcurve:

Visualize the folded lightcurve based on the discovered period.
![image](https://github.com/ShreyasTembhare/Exoplanet-detector/assets/77243055/dc36d175-aafc-421c-987d-5835162ce8d2)
![image](https://github.com/ShreyasTembhare/Exoplanet-detector/assets/77243055/77666d57-fb87-4bcc-b372-72b645b7b671)


Downloader Helper:
The downloader helper script allows you to fetch additional data from MAST for a specific product group ID. To use this helper:

Update the product_group_id variable with the desired product group ID.
Set the url variable with the appropriate MAST API link.
Specify the destination folder where you want to save the downloaded data.
Run the code to download and unzip the data.
