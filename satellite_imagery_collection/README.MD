This folder contains scripts for 

### generating image tile numbers according to the city boundary files: 

make_tile_scd.py,

### downloading Esri images: 

download_image.py,


### Note on Cities 
Census tract count per city
New York city        4217
Los Angeles city     1993
Chicago city         1894
Philadelphia city     942
San Antonio city      826
Phoenix city          819
Houston city          729
San Diego city        692
Jacksonville city     314

Smallest photos per census tract
Min photos for New York: 4:360050019041
Min photos for Los Angeles: 4:60372063022
Min photos for Chicago: 4:170310301042
Min photos for Houston: 4:482012319004
Min photos for Phoenix: 4:40131036123
Min photos for Philadelphia: 4:421010007012
Min photos for San Antonio: 4:480291514002
Min photos for San Diego: 4:60730031051
Min photos for Jacksonville: 48:120310144201 

Max photos for New York: 38808:360819901000
Max photos for Los Angeles: 14596:60379902000
Max photos for Chicago: 6640:170318388002
Max photos for Houston: 7232:482015432022
Max photos for Phoenix: 15696:40131167331
Max photos for Philadelphia: 1716:421019800021
Max photos for San Antonio: 6536:480291612001
Max photos for San Diego: 22116:60739901000
Max photos for Jacksonville: 27452:120310173003

There are 559 distinct CBGs in New York with more than 20 rows.
There are 846 distinct CBGs in Los Angeles with more than 20 rows.
There are 559 distinct CBGs in Chicago with more than 20 rows.
There are 587 distinct CBGs in Dallas with more than 20 rows.
There are 1240 distinct CBGs in Houston with more than 20 rows.
There are 611 distinct CBGs in Phoenix with more than 20 rows.
There are 309 distinct CBGs in Philadelphia with more than 20 rows.
There are 565 distinct CBGs in San Antonio with more than 20 rows.
There are 283 distinct CBGs in San Diego with more than 20 rows.
There are 536 distinct CBGs in Jacksonville with more than 20 rows.

6095 total tracts

Leave out LA: test is 13.8%
Remainder 5249 I want 18.8% of these to go to val
