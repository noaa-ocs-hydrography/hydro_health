#-------------------------------------------------------------------------------------------------------
#----------------		Creating standardised rasters 		----------------
#-------------------------------------------------------------------------------------------------------


# Watson Notes# 
# can use this script to loop through all the layer files from GIS instead of manually clipping them all,
# once we have boundary defined and mask file created. 

# there is also script information for pre/post processing here https://cbig.github.io/zonator/articles/zonator-project.html (however the package is discontinued)


require(raster) 
# require(rgdal) # previously written with rgdal but now superseded 
require(sf)
require(terra)
require(sp)
require(stringr)

# EXTRACT BY MASK, and make sure all same extent ----
       
#------------------------------------	import raster mask
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model") # directory with your mask file 
mask <- raster("pilot_model_mask.tif")  # my file must be named to match. clips raster files to mask boundary
plot(mask)
mask # view dimensions -MASK--
# mask[mask < 0.5] <- NA
# mask[mask >= 0.5] <- 1
# 

mask.df <- as.data.frame(rasterToPoints(mask)) # extract x y coords for mask raster
mask.df <- mask.df[,-3] # creates a seperate data frame with esting and northing. 
names(mask.df) <- c("Easting","Northing")
View(mask.df) # you can see all of your easting and northing coordinates here 

crs <- crs(mask) # make sure using right projection

#------------------------------------	load files to be clipped
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/training/raw")
f250m <- list.files(getwd())
ras250m <- lapply(f250m, raster) # won't work if any other files other than .tif are in the directory / folder
plot(ras250m[[1]]) # check all are on the same units and projection #changing number in brackets selects files


 
# this subset should match how ever many tiff files your have in your directory.
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/training/out") # change this directry to your output FOLDER! 



for (i in 1:length(ras250m)){ #extract
  temp.df <- cbind(mask.df, as.data.frame(extract(ras250m[[i]], mask.df)))
 # temp.df[is.na(temp.df)] <- 0.0001 # give a small value so that all have same area with data
  temp.ras <- rasterFromXYZ(temp.df, crs = crs)
  # layer nam
  name <- paste(names(ras250m[[i]]), "clip", sep = "_")
  assign(name, temp.ras, envir = .GlobalEnv)
  writeRaster(temp.ras, filename = paste(names(ras250m[[i]]), ".tif", sep = ""), overwrite=T)
  }

                                          ######   END #####

# MODIFY FILE NAME----
# required(stringr) # modify text strings 
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Bathy_below_0/2004_PostIvan_Job1051878")

f250m <- list.files(getwd())

file.rename(list.files(pattern ='_prj'),
            str_replace(list.files(pattern='_prj'), pattern='_prj', '')) # change out the last pattern of unwanted text name




######--------CROP or EXTEND-------### 
setwd("C:/Users/sw277/Desktop/R/PhD/Chap3/R_working/out") # change this directry to your output FOLDER! 


# CROP ----

for (i in 1:length(ras250m)){
  temp.df <- cbind(mask.df, as.data.frame(crop(ras250m[[i]], mask.df)))
 # temp.df[is.na(temp.df)] <- 0.0001 # give a small value so that all have same area with data
  temp.ras <- rasterFromXYZ(temp.df, crs = crs)
  # layer nam
  name <- paste(names(ras250m[[i]]), "clip", sep = "_")
  assign(name, temp.ras, envir = .GlobalEnv)
  writeRaster(temp.ras, filename = paste(names(ras250m[[i]]), ".tif", sep = ""), overwrite=T)
}

setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model") # directory with your mask file 
mask <- raster("pilot_model_mask.tif")  # my file must be named to match. clips raster files to mask boundary
plot(mask)
mask # view dimensions -MASK--
# mask[mask < 0.5] <- NA
# mask[mask >= 0.5] <- 1
# 

mask.df <- as.data.frame(rasterToPoints(mask)) # extract x y coords for mask raster
mask.df <- mask.df[,-3] # creates a seperate data frame with esting and northing. 
names(mask.df) <- c("Easting","Northing")
View(mask.df) # you can see all of your easting and northing coordinates here 

crs <- crs(mask) # make sure using right projection


setwd("C:/Users/sw277/Desktop/R/PhD/Chap3/Env_preds_2021_to_extend")
# EXTEND----
for (i in 1:length(ras250m)){
  temp.df <- (extend(ras250m[[i]], mask))
  # temp.df[is.na(temp.df)] <- 0.0001 # give a small value so that all have same area with data
  name <- paste(names(ras250m[[i]]), "clip", sep = "_")
  assign(name, temp.df, envir = .GlobalEnv)
  writeRaster(temp.df, filename = paste(names(ras250m[[i]]), ".tif", sep = ""), overwrite=T)
}










test <- crop(ras250m[[1]], mask)
plot(test)




a <- crop(ras250m[[3]], mask)
plot(a)

ras250m[[5]]
mask

a <-extend(ras250m[[1]], mask) 
b <- extend(ras250m[[2]], mask)
c <- extend(ras250m[[3]], mask)
d <- extend(ras250m[[4]], mask)
e <- extend(ras250m[[5]], mask)

extend(ras250m[[2]], mask)
extend(ras250m[[3]], mask)
extend(ras250m[[4]], mask)

plot(b)


