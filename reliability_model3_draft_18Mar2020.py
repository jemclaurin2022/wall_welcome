# -*- coding: utf-8 -*-
"""
********************************************************************************************************************
Purpose: Draft version of code for fragility curve calculations.
This code is a work in progress, likely to change, and is provided only for integration planning
This code has not undergone QA testing and the outputs are not suitable for use in reliability or risk analyses
PG&E assumes any risk associated with use of this code 
********************************************************************************************************************
Key outputs:
AGE_YEARS - Age of the asset or component based on the current year minus the
            the installation year from Eszter's data.
design_life_adjustment - Calculated adjustment factor for the design life of a
            particular asset or component. This factor is multiplied by the
            design life as specified in df_reliability_calcs_constants to get
            the adjusted design life (design_life_adjusted).
design_life_adjusted - Calculated adjusted design life based on the
            design_life_adjustment multiplied by the design life specified in
            df_reliability_calcs_constants.
cov - Coeffient of variation calculated for each component.
strength_ratio - Strength ratio calculated for each component based on the
            corresponding Pronto codes.
design_ratio - Design ratio calculated for each component.
p_f_at_wspeed - Probability of failure (p_f) values calculated at specified
                windspeed increments (wspeed) from 0 miles per hour (mph) up to
                at least 120 mph. At present, wspeed is set to 1 mph.
df0 - DataFrame output of compiled inputs and calculations.
df_times_performance - DataFrame output of time (in seconds) to complete key
            steps in the analysis such as assembling the user input values,
            reading in the input csv files as DataFrames, calculating values
            such as design life adjusted and cov, and calculate the probability
            of failure at all of the desired windspeed.

@author: jglassman, egroves, skothari-phan
"""

import pandas as pd
import os
import numpy as np
import math
from scipy.stats import norm
from scipy.stats import lognorm
import datetime
import parameters # Module of hard-coded values not pulled from database
import config # Configuration file with Exponent database info
import pyodbc
import urllib
from sqlalchemy import create_engine
from sqlalchemy.dialects.mssql import DECIMAL, VARCHAR, DATETIME, INTEGER, FLOAT

#-----------------------------------------------------------------------------#
#                               FUNCTIONS                                     #
#-----------------------------------------------------------------------------#

def ComputeProbabilityFailureLogNorm(wspeed, mean_ANCHOR, stddev_ANCHOR, mean_GUY,stddev_GUY,mean_FOUNDATION, 
                                     stddev_FOUNDATION, mean_STUB_SPLICE, 
                                     stddev_STUB_SPLICE, mean_STRUCT_ATTACH, stddev_STRUCT_ATTACH, mean_CONDUCTOR, 
                                     stddev_CONDUCTOR, mean_OGW, stddev_OGW, mean_HI, stddev_HI):

    #*******************************************************************#
    # Purpose: To calculate the probability of failure at the specified #
    #          windspeed.                                               #
    #*******************************************************************#

    prob_fail = ((1 - (1 - lognorm.cdf(wspeed, mean_ANCHOR, stddev_ANCHOR)) * \
                  (1 - lognorm.cdf(wspeed, mean_GUY, stddev_GUY)) * \
                  (1 - lognorm.cdf(wspeed, mean_FOUNDATION, stddev_FOUNDATION)) * \
                  (1 - lognorm.cdf(wspeed, mean_STUB_SPLICE, stddev_STUB_SPLICE)) * \
                  (1 - lognorm.cdf(wspeed, mean_STRUCT_ATTACH, stddev_STRUCT_ATTACH)) * \
                  (1 - lognorm.cdf(wspeed, mean_CONDUCTOR, stddev_CONDUCTOR)) * \
                  (1 - lognorm.cdf(wspeed, mean_OGW, stddev_OGW)) * \
                  (1 - lognorm.cdf(wspeed, mean_HI, stddev_HI))) + \
                  np.maximum.reduce([lognorm.cdf(wspeed, mean_ANCHOR, stddev_ANCHOR), \
                     lognorm.cdf(wspeed, mean_GUY, stddev_GUY), \
                     lognorm.cdf(wspeed, mean_FOUNDATION, stddev_FOUNDATION), \
                     lognorm.cdf(wspeed, mean_STUB_SPLICE, stddev_STUB_SPLICE), \
                     lognorm.cdf(wspeed, mean_STRUCT_ATTACH, stddev_STRUCT_ATTACH), \
                     lognorm.cdf(wspeed, mean_CONDUCTOR, stddev_CONDUCTOR), \
                     lognorm.cdf(wspeed, mean_OGW, stddev_OGW), \
                     lognorm.cdf(wspeed, mean_HI, stddev_HI)])) / 2

    return prob_fail


def DLife_WoodSteel_Conductor(AGE_YEARS,df_reliability_calcs_constants,pronto,outage_density_red_factor,wear_fatigue_red_factor,splice_density_red_factor,atmospheric_corrosivity_red_factor):
    #********************************************************************#
    # Purpose: To calculate the design life adjustment, adjusted design  #
    #          life, and cov for the CONDUCTOR pronto theme. This same   #
    #          function can be used for STEEL and NONSTEEL structures.   #
    #          DLife is short for 'design life'.                         #
    #********************************************************************#
    # Calculate the adjustment factor the design life:
    if outage_density_red_factor < 0:
        design_life_adjustment = (1-math.sqrt(wear_fatigue_red_factor**2 + splice_density_red_factor**2 + atmospheric_corrosivity_red_factor**2))*(1-outage_density_red_factor)
    else:
        design_life_adjustment = 1-math.sqrt(wear_fatigue_red_factor**2 + splice_density_red_factor**2 + atmospheric_corrosivity_red_factor**2 + outage_density_red_factor**2)

    # Calculate the adjusted design life:
    design_life_adjusted = df_reliability_calcs_constants.loc[2,pronto]*design_life_adjustment

    # Calculate the coefficient of variation:
    cov = df_reliability_calcs_constants.loc[0,pronto] + (df_reliability_calcs_constants.loc[1,pronto] - df_reliability_calcs_constants.loc[0,pronto])*(AGE_YEARS**2/design_life_adjusted**2)

    return design_life_adjustment, design_life_adjusted, cov


def DLife_WoodSteel_Anchor(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,soil_corrosivity_red_factor):
    #********************************************************************#
    # Purpose: To calculate the design life adjustment, adjusted design  #
    #          life, and cov for the ANCHOR pronto theme. This same      #
    #          function can be used for STEEL and NONSTEEL structures.   #
    #          DLife is short for 'design life'.                         #
    #********************************************************************#
    if outage_density_red_factor < 0:
        design_life_adjustment = (1-soil_corrosivity_red_factor)*(1-outage_density_red_factor)
    else:
        design_life_adjustment = 1-math.sqrt(soil_corrosivity_red_factor**2 + outage_density_red_factor**2)

    # Calculate adjusted design life:
    design_life_adjusted = df_reliability_calcs_constants.loc[2,entry]*design_life_adjustment

    # Calculate cov for the component
    cov = df_reliability_calcs_constants.loc[0,entry] + (df_reliability_calcs_constants.loc[1,entry] - df_reliability_calcs_constants.loc[0,entry])*(AGE_YEARS**2/design_life_adjusted**2)

    return design_life_adjustment, design_life_adjusted, cov

def DLife_WoodSteel_Guy(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor):
    #********************************************************************#
    # Purpose: To calculate the design life adjustment, adjusted design  #
    #          life, and cov for the GUY pronto theme. This same         #
    #          function can be used for STEEL and NONSTEEL structures.   #
    #          DLife is short for 'design life'.                         #
    #********************************************************************#
    if outage_density_red_factor < 0:
        design_life_adjustment = (1-atmospheric_corrosivity_red_factor)*(1-outage_density_red_factor)
    else:
        design_life_adjustment = 1-math.sqrt(atmospheric_corrosivity_red_factor**2 + outage_density_red_factor**2)

    # Calculate adjusted design life:
    design_life_adjusted = df_reliability_calcs_constants.loc[2,entry]*design_life_adjustment

    # Calculate cov for the component:
    cov = df_reliability_calcs_constants.loc[0,entry] + (df_reliability_calcs_constants.loc[1,entry] - df_reliability_calcs_constants.loc[0,entry])*(AGE_YEARS**2/design_life_adjusted**2)

    return design_life_adjustment, design_life_adjusted, cov

def DLife_WoodSteel_OGW(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor):
    #********************************************************************#
    # Purpose: To calculate the design life adjustment, adjusted design  #
    #          life, and cov for the Overhead Guy Wire (OGW) pronto      #
    #          theme. This same function can be used for STEEL and       #
    #          NONSTEEL structures.                                      #
    #          DLife is short for 'design life'.                         #
    #********************************************************************#
    if outage_density_red_factor < 0:
        design_life_adjustment = 1-math.sqrt(soil_corrosivity_red_factor**2 + atmospheric_corrosivity_red_factor**2)*(1-outage_density_red_factor)
    else:
        design_life_adjustment = 1-math.sqrt(soil_corrosivity_red_factor**2 + atmospheric_corrosivity_red_factor**2 + outage_density_red_factor**2)

    # Calculate adjusted design life:
    design_life_adjusted = df_reliability_calcs_constants.loc[2,entry]*design_life_adjustment

    # Calculate cov for the component
    cov = df_reliability_calcs_constants.loc[0,entry] + (df_reliability_calcs_constants.loc[1,entry] - df_reliability_calcs_constants.loc[0,entry])*(AGE_YEARS**2/design_life_adjusted**2)

    return design_life_adjustment, design_life_adjusted, cov

def DLife_WoodSteel_HI(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor):
    #********************************************************************#
    # Purpose: To calculate the design life adjustment, adjusted design  #
    #          life, and cov for the HARDWARE_INSULATORS pronto theme.   #
    #          This same function can be used for STEEL and NONSTEEL     #
    #           structures.                                              #
    #          DLife is short for 'design life'.                         #
    #********************************************************************#
    if outage_density_red_factor < 0:
        design_life_adjustment = 1-math.sqrt(soil_corrosivity_red_factor**2 + atmospheric_corrosivity_red_factor**2)*(1-outage_density_red_factor)
    else:
        design_life_adjustment = 1-math.sqrt(soil_corrosivity_red_factor**2 + atmospheric_corrosivity_red_factor**2 + outage_density_red_factor**2)

    # Calculate adjusted design life:
    design_life_adjusted = df_reliability_calcs_constants.loc[2,entry]*design_life_adjustment

    # Calculate cov for the component
    cov = df_reliability_calcs_constants.loc[0,entry] + (df_reliability_calcs_constants.loc[1,entry] - df_reliability_calcs_constants.loc[0,entry])*(AGE_YEARS**2/design_life_adjusted**2)

    return design_life_adjustment, design_life_adjusted, cov

def DLife_WoodSteel_StructureFoundation(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor, soil_corrosivity_red_factor):
    #********************************************************************#
    # Purpose: To calculate the design life adjustment, adjusted design  #
    #          life, and cov for the STRUCTURE (for WOOD structures) and #
    #          FOUNDATION (for STEEL structures) pronto theme. This same #
    #          function can be used for STEEL and NONSTEEL structures.   #
    #          DLife is short for 'design life'.                         #
    #********************************************************************#
    if outage_density_red_factor < 0:
        design_life_adjustment = (1-math.sqrt(wear_fatigue_red_factor**2 + soil_corrosivity_red_factor**2 + atmospheric_corrosivity_red_factor**2))*(1-outage_density_red_factor)
    else:
        design_life_adjustment = 1-math.sqrt(wear_fatigue_red_factor**2 + soil_corrosivity_red_factor**2 + atmospheric_corrosivity_red_factor**2 + outage_density_red_factor**2)

    # Calculate adjusted design life:
    design_life_adjusted = df_reliability_calcs_constants.loc[2,entry]*design_life_adjustment

    # Calculate cov for the component
#    cov = cov_steel+(cov_D_steel-cov_steel)*(AGE_YEARS**2/design_life_adjusted**2)
    cov = df_reliability_calcs_constants.loc[0,entry] +(df_reliability_calcs_constants.loc[1,entry]-df_reliability_calcs_constants.loc[0,entry])*(AGE_YEARS**2/design_life_adjusted**2)

    return design_life_adjustment, design_life_adjusted, cov

def DLife_WoodSteel_AllOthers(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor):
    #********************************************************************#
    # Purpose: To calculate the design life adjustment, adjusted design  #
    #          life, and cov for the remaining pronto themes. This same  #
    #          function can be used for STEEL and NONSTEEL structures.   #
    #          DLife is short for 'design life'.                         #
    #********************************************************************#
    if outage_density_red_factor < 0:
            design_life_adjustment = (1-math.sqrt(wear_fatigue_red_factor**2 + atmospheric_corrosivity_red_factor**2))*(1-outage_density_red_factor)
    else:
        design_life_adjustment = 1-math.sqrt(wear_fatigue_red_factor**2 + atmospheric_corrosivity_red_factor**2 + outage_density_red_factor**2)

    # Calculate adjusted design life:
    design_life_adjusted = df_reliability_calcs_constants.loc[2,entry]*design_life_adjustment

    # Calculate cov for the component
#    cov = cov_steel+(cov_D_steel-cov_steel)*(AGE_YEARS**2/design_life_adjusted**2)
    cov = df_reliability_calcs_constants.loc[0,entry] +(df_reliability_calcs_constants.loc[1,entry]-df_reliability_calcs_constants.loc[0,entry])*(AGE_YEARS**2/design_life_adjusted**2)

    return design_life_adjustment, design_life_adjusted, cov

startTime = datetime.datetime.now()
#-----------------------------------------------------------------------------#
#                                  INPUT                                      #
#-----------------------------------------------------------------------------#
# Flags for writing to file or database
writeCSV = True
writeDB = False

filename_for_Bayesian_delta_medians = 'Bayesian_DeltaMedians_10202019.csv' # Input filename for delta medians values from Bayesian updating (at ETL level).
#-----------------------------------------------------------------------------#
#                             DATABASE IMPORT                                 #
#-----------------------------------------------------------------------------#
# Database import:


conn = pyodbc.connect(driver="{SQL Server}", server=config.ExpoServer, database=config.ExpoDatabase, trusted_connection='yes')

SQL_Query = pd.read_sql_query(
    '''SELECT 
        sD.SAP_EQUIP_ID, sD.ETGIS_ID, sD.STRUCTURE_NO, sD.WEAR_FATIGUE_RED_FAC, sD.SAP_FUNC_LOC_NO, sD.AGRICULTURE, sD.WETLAND_TYPE, sD.CORROSION_ZONE, sD.INSTALLED_YEAR, sD.MATERIAL_FLAG, sD.ANCHOR_CD, sD.GUY_CD, sD.STRUCTURE_CD, sD.FOUNDATION_CD, sD.CROSSARMS_CD, sD.FRAME_ATTACH_CD, sD.STRUCT_ATTACH_CD, sD.STUB_SPLICE_CD, sD.CONDUCTOR_CD, sD.OGW_CD, sD.HARDWARE_INSUL_CD, 

        sD.WSIP_SCOPE_IND, 
        sD.HOST_TLINE_NM,
        sD.SPLICES,

        tD.TLINE_MILES, tD.OUTAGE_DESIGNLIFE_MOD

      FROM [PGE_OA].[dbo].[CSV_Structure] sD
      INNER JOIN [PGE_OA].[dbo].[CSV_TLine] tD on sD.SAP_FUNC_LOC_NO=tD.SAP_FUNC_LOC_NO 
    ''', conn)

df0 = pd.DataFrame(SQL_Query)

# Time indexing purposes:
df_times = []   # Initialize a list to store reported times (in seconds) at key steps in the script.
# MCE corrosion scores DataFrame (indexed to ROW_LABELS):
# (Needs updating) Hard-coded values below will be read from the database tables.
# (Needs updating) Manually added "Vacant or Disturbed Land" and "Rural Residential Land"  and 'Semi-agricultural and Rural Commercial Land' and 'Farmland of Local Potential', 'Water Area', with agriculture scores 0 for testing.

df_MCE_corrosion_scores = pd.DataFrame()  # Initialize the Pandas DataFrame.
df_MCE_corrosion_scores['ROW_LABELS'] = ['heavy', 'intermediate', 'light', 'Estuarine and Marine Deepwater',
                                         'Estuarine and Marine Wetland', 'Freshwater Emergent Wetland',
                                         'Freshwater Forested/Shrub Wetland', 'Freshwater Pond', 'Lake', 'Riverine',
                                         'other', 'Blank', 'None', 'Farmland of Local Importance',
                                         'Farmland of Statewide Importance', 'Grazing Land',
                                         'Irrigated Farmland (interim)', 'Local Potential',
                                         'Nonagricultural and Natural Vegetation', 'Not Mapped', 'Other Land',
                                         'Prime Farmland', 'Rural Residential and Rural Commercial',
                                         'Unique Farmland',
                                         'Urban and Built-up Land', 'Water', 'Confined Animal Agriculture',
                                         'moderate',
                                         'severe', 'Lower 1/3 of PG&E wind speed', 'Middle 1/3 of PG&E wind speed',
                                         'Top 1/3 of PG&E wind speed', 'High', 'Med', 'Low',
                                         'Vacant or Disturbed Land', 'Rural Residential Land', 'Farmland of Local Potential','Semi-agricultural and Rural Commercial Land', 'Water Area']
df_MCE_corrosion_scores['SNOWLOAD'] = [2, 1, 0, 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                       'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A','ERROR_MCE_N/A','ERROR_MCE_N/A']
df_MCE_corrosion_scores['WETLAND_TYPE'] = ['ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 2, 2, 1, 1, 1, 1, 1,
                                           1, 0,
                                           0, 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                           'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                           'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                           'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                           'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                           'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A','ERROR_MCE_N/A','ERROR_MCE_N/A']
df_MCE_corrosion_scores['AGRICULTURE'] = ['ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                          'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                          'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                          'ERROR_MCE_N/A', 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 0, 2, 2,
                                          'ERROR_MCE_N/A',
                                          'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                          'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 0, 0, 0, 0, 0]
df_MCE_corrosion_scores['ATMOSPHERIC_CORROSION'] = ['ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A',
                                                    0, 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 1, 2,
                                                    'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A',
                                                    'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A']
df_MCE_corrosion_scores['WIND_SPEED'] = ['ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 0, 1, 2, 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A',
                                         'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A']
df_MCE_corrosion_scores['SOILS_RESISTIVITY'] = ['ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A',
                                                0,
                                                1, 2, 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A', 'ERROR_MCE_N/A']
df_MCE_corrosion_scores_indexed = df_MCE_corrosion_scores.set_index('ROW_LABELS')

# Assemble a pandas DataFrame of user-defined component-based constants:
#*********************************************************************#
# Note: I label the columns the same as the Pronto code labels in     #
#       Eszter's data output so that as we run calculations component #
#       by component (cf. 'for entry in steel_pronto_themes:'         #
#       multiple lines of code below this) we can use the same        #
#       'entry' value in the for loop to locate the corresponding     #
#       constant values in the df_reliability_calcs_constants.        #
#*********************************************************************#
df_reliability_calcs_constants = pd.DataFrame() # Initialize the Pandas DataFrame to hold all of the component-based constants.
df_reliability_calcs_constants['ANCHOR_CD'] = parameters.anchor_constants
df_reliability_calcs_constants['GUY_CD'] = parameters.guy_constants
df_reliability_calcs_constants['FOUNDATION_CD'] = parameters.foundation_constants
df_reliability_calcs_constants['STUB_SPLICE_CD'] = parameters.stub_splice_constants
df_reliability_calcs_constants['FRAME_ATTACH_CD'] = parameters.framing_attachments_constants
df_reliability_calcs_constants['STRUCT_ATTACH_CD'] = parameters.structure_attachments_constants
df_reliability_calcs_constants['CONDUCTOR_CD'] = parameters.conductor_constants
df_reliability_calcs_constants['OGW_CD'] = parameters.ogw_constants
df_reliability_calcs_constants['HARDWARE_INSUL_CD'] = parameters.hardware_insulators_constants
df_reliability_calcs_constants['STRUCTURE_CD'] = parameters.structure_constants
df_reliability_calcs_constants['CROSSARMS_CD'] = parameters.crossarms_constants

print("Time spent for processing and assembling user-input calculations:",datetime.datetime.now() - startTime,"------------")
df_times.append((datetime.datetime.now() - startTime).total_seconds())

#-----------------------------------------------------------------------------#
#                              MAIN CODE                                      #
#-----------------------------------------------------------------------------#
now = datetime.datetime.now() # Identify the current date-time-year.

#-------------------------------------------#
# Begin Pronto-theme dependent calculations #
#-------------------------------------------#
for entry in parameters.steel_pronto_themes:
    design_life_adjustment_values = []              # Design life adjustment values calculated based on the Pronto theme under consideration.
    design_life_adjusted_values = []                # Adjusted design life running list, based on calculations with the design life adjustment values.
    strength_ratio_values = []                      # Strength ratio running list of calculated values.
    design_ratio_values = []                        # Design ratio running list of calculated values.
    cov_values = []                                 # Coefficient of variation running list of calculated values.
    prob_fail_forecast_wind_values = []             # Probability of failure at the forecast windspeed (may become obsolete as of 6/10/2019 discussion with Will).
    wear_fatigue_red_factor_values = []             # Wear and fatigue reduction factors running list of calculated values.
    score_agriculture_values = []                   # Agriculture score running list of calculated values.
    score_wetland_values = []                       # Wetland score running list of calculated values.
    score_atmospheric_corrosion_values = []         # Atmospheric score running list of calculated values.
    outage_density_red_factor_values = []           # Outage density reduction factor running list of calculated values.
    soil_corrosivity_red_factor_values = []         # Soil corrosivity reduction factor running list of calculated values.
    atmospheric_corrosivity_red_factor_values = []  # Atmospheric corrosivity reduction factor running list of calculated values.
    splice_count_values = []                        # Splice count running list of pre-calculated values.
    splice_density_red_factor_values = []           # Splice density reduction factor running list of calculated values.

    for p in range(0,np.shape(df0)[0]):
        #---------------------------------------------------------------------#
        #               Begin structure-dependent calculations                #
        #---------------------------------------------------------------------#
        # Structure-dependent calculations are for:                           #
        # 1. outage_density_red_factor                                        #
        # 2. splice_density_red_factor                                        #
        # 3. soil_corrosivity_red_factor                                      #
        # 4. atmospheric_corrosivity_red_factor                               #
        # 5. agriculture score                                                #
        # 6. wetland score                                                    #
        #*********************************************************************#

        outage_density_red_factor = -df0.loc[p,'OUTAGE_DESIGNLIFE_MOD']

        # Splice density reduction factor calculation:
        splice_density_red_factor = min(df0.loc[p,'SPLICES']/5*parameters.r_spl,parameters.r_spl)

        # Wear and fatigue reduction factor calculation:
        wear_fatigue_red_factor = df0.loc[p,'WEAR_FATIGUE_RED_FAC']

       # Agriculture score calculation:
        classification_agriculture = df0.loc[p,'AGRICULTURE']                                               # Agriculture classification type from input data
        score_agriculture = df_MCE_corrosion_scores_indexed.loc[classification_agriculture,'AGRICULTURE']   # Agriculture score (Exponent calculation)

        # Wetland score calculation:
        classification_wetland = df0.loc[p,'WETLAND_TYPE']                                                  # Wetland classification type from input data
        score_wetland = df_MCE_corrosion_scores_indexed.loc[classification_wetland,'WETLAND_TYPE']          # Wetland score (Exponent calculation)

        # Atmospheric corrosion score calculation:
        classification_corrosion_atmospheric = df0.loc[p,'CORROSION_ZONE']                                  # Atmospheric corrosion classification type from input data
        score_corrosion_atmospheric = df_MCE_corrosion_scores_indexed.loc[classification_corrosion_atmospheric,'ATMOSPHERIC_CORROSION'] # Atmospheric corrosion score (Exponent calculation)

        # Soil corrosivity reduction factor calculation:
        soil_corrosivity_red_factor = float(max(score_agriculture,score_wetland))/2*parameters.r_cor                   # Soil corrosivity reduction factor
        if math.isnan(soil_corrosivity_red_factor) is True:                                                 # Set Nan values to 0
            soil_corrosivity_red_factor = 0

        # Atmospheric corrosivity reduction factor calculation:
        atmospheric_corrosivity_red_factor = float(max(score_wetland,score_corrosion_atmospheric))/2*parameters.r_cor  # Atmospheric corrosivity reduction factor
        if math.isnan(atmospheric_corrosivity_red_factor) is True:                                          # Set Nan values to 0
            atmospheric_corrosivity_red_factor = 0

        # Append structure-dependent calculations to the respective running list:
        wear_fatigue_red_factor_values.append(wear_fatigue_red_factor)
        score_agriculture_values.append(score_agriculture)
        score_wetland_values.append(score_wetland)
        score_atmospheric_corrosion_values.append(score_corrosion_atmospheric)
        outage_density_red_factor_values.append(outage_density_red_factor)
        soil_corrosivity_red_factor_values.append(soil_corrosivity_red_factor)
        atmospheric_corrosivity_red_factor_values.append(atmospheric_corrosivity_red_factor)
        #splice_count_values.append(splice_count)
        #splice_density_red_factor_values.append(splice_density_red_factor)

        #---------------------------------------------------------------------#
        #        Begin material-dependent calculations (per structure)        #
        #---------------------------------------------------------------------#
        # Material- and component-dependent calculations are for:             #
        # 1. design life adjustment                                           #
        # 2. design life adjusted                                             #
        # 3. age in years                                                     #
        # 4. strength ratio                                                   #
        # 5. design ratio                                                     #
        # 6. coefficient of variation (cov)                                   #
        # 7. probability of failure at forecast wind speed                    #
        #*********************************************************************#
        # Determine if structure is STEEL, WOOD, or UNKNOWN or OTHER.         #
        # Notes:                                                              #
        # 1. In Excel model we may only distinguish between STEEL structures  #
        # (STEEL) and NONSTEEL structures (WOOD, UNKNOWN, OTHER).             #
        # 2. AGE_YEARS is calculated for every Pronto theme ('CONDUCTOR_CD,   #
        # ANCHOR_CD, etc.) because eventually each of these components will   #
        # have a separate age attached to them not the current age of the     #
        # structure that we are currently using.                              #
        #*********************************************************************#
        if df0.loc[p,'MATERIAL_FLAG'] == 'STEEL':
            # Calculate design life adjustment value:
            if entry == 'CONDUCTOR_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Conductor(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,wear_fatigue_red_factor,splice_density_red_factor,atmospheric_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'ANCHOR_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Anchor(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'GUY_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Guy(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'OGW_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_OGW(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'HARDWARE_INSUL_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_HI(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'FOUNDATION_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_StructureFoundation(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            else: # For CROSSARMS_CD, STUB_SPLICE_CD, FRAME_ATTACH_CD, STRUCT_ATTACH_CD
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_AllOthers(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            design_life_adjustment_values.append(design_life_adjustment)

            # Calculate the strength ratio:
            if df0.loc[p,entry] == 0 or math.isnan(df0.loc[p,entry]) is True:
                strength_ratio = 1
            elif df0.loc[p,entry] == 2:
                strength_ratio = 0.92
            else:
                strength_ratio = 1 - ((df0.loc[p,entry]-1)/6)
            strength_ratio_values.append(strength_ratio)

            # Calculate the design ratio:
            design_ratio = 1
            design_ratio_values.append(design_ratio)

            #print("Additional STEEL calculations for structure entry number: ", p, " of", np.shape(df0)[0], " has been processed for component: ", entry,'at:', datetime.datetime.now() - startTime,"------------")

        elif df0.loc[p,'MATERIAL_FLAG'] == 'WOOD':
            # Rename entry name to entry_wood based upon the following criteria:
            #*****************************************************************#
            # Note: This rename step makes sure that the correct Pronto theme #
            #       is referenced from Eszter's data.                         #
            # FOUNDATION_CD for STEEL structures is STRUCTURE_CD for NONSTEEL #
            # STRUCT_ATTACH_CD for STEEL structures is FRAME_ATTACH_CD for    #
            #                                                        NONSTEEL #
            # STUB_SPLICE_CD for STEEL structures is CROSSARMS_CD for NONSTEEL#
            #*****************************************************************#
            if entry == 'FOUNDATION_CD':
                entry_wood = 'STRUCTURE_CD'
            elif entry == 'STRUCT_ATTACH_CD':
                entry_wood = 'FRAME_ATTACH_CD'
            elif entry == 'STUB_SPLICE_CD':
                entry_wood = 'CROSSARMS_CD'
            else:
                entry_wood = entry

            # Calculate design life adjustment value:
            if entry_wood == 'CONDUCTOR_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Conductor(AGE_YEARS,df_reliability_calcs_constants,entry_wood,outage_density_red_factor,wear_fatigue_red_factor,splice_density_red_factor,atmospheric_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry_wood == 'ANCHOR_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Anchor(AGE_YEARS,df_reliability_calcs_constants,entry_wood,outage_density_red_factor,soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry_wood == 'GUY_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Guy(AGE_YEARS,df_reliability_calcs_constants,entry_wood,outage_density_red_factor,atmospheric_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry_wood == 'OGW_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_OGW(AGE_YEARS,df_reliability_calcs_constants,entry_wood,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry_wood == 'HARDWARE_INSUL_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_HI(AGE_YEARS,df_reliability_calcs_constants,entry_wood,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry_wood == 'STRUCTURE_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_StructureFoundation(AGE_YEARS,df_reliability_calcs_constants,entry_wood,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            else: # For CROSSARMS_CD, STUB_SPLICE_CD, FRAME_ATTACH_CD, STRUCT_ATTACH_CD
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_AllOthers(AGE_YEARS,df_reliability_calcs_constants,entry_wood,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            design_life_adjustment_values.append(design_life_adjustment)

            # Calculate the strength ratio:
            if df0.loc[p,entry_wood] == 0 or math.isnan(df0.loc[p,entry_wood]) is True:
                strength_ratio = 1
            elif df0.loc[p,entry_wood] == 2:
                strength_ratio = 0.92
            else:
                strength_ratio = 1 - ((df0.loc[p,entry_wood]-1)/6)
            strength_ratio_values.append(strength_ratio)

            # Calculate the design ratio:
            design_ratio = 1
            design_ratio_values.append(design_ratio)

           # print("Additional WOOD calculations for structure entry number: ", p, " of", np.shape(df0)[0], " has been processed for component: ", entry_wood,'at', datetime.datetime.now() - startTime,"------------")
        else:
            #**************************************************************************#
            # CODE CHECK:                                                              #
            # UNKNOWN and OTHER material types calculated as STEEL for now.            #
            #**************************************************************************#
            # Calculate design life adjustment value:
            if entry == 'CONDUCTOR_CD':

                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Conductor(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,wear_fatigue_red_factor,splice_density_red_factor,atmospheric_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'ANCHOR_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Anchor(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'GUY':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_Guy(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'OGW_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_OGW(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'FOUNDATION_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_StructureFoundation(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            elif entry == 'HARDWARE_INSUL_CD':
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_HI(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, soil_corrosivity_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            else: # For CROSSARMS_CD, STUB_SPLICE_CD, FRAME_ATTACH_CD, STRUCT_ATTACH_CD
                # Calculate age of component (in years):
                AGE_YEARS = now.year - df0.loc[p,'INSTALLED_YEAR']

                # Calculate design life adjustment, adjusted design life, and cov:
                design_life_adjustment, design_life_adjusted, cov = DLife_WoodSteel_AllOthers(AGE_YEARS,df_reliability_calcs_constants,entry,outage_density_red_factor,atmospheric_corrosivity_red_factor, wear_fatigue_red_factor)
                design_life_adjusted_values.append(design_life_adjusted)
                cov_values.append(cov)

            # Append calculated design life adjustment to the list for documentation:
            design_life_adjustment_values.append(design_life_adjustment)

            # Calculate the strength ratio:
            if df0.loc[p,entry] == 0 or math.isnan(df0.loc[p,entry]) is True:
                strength_ratio = 1
            elif df0.loc[p,entry] == 2:
                strength_ratio = 0.92
            else:
                strength_ratio = 1 - ((df0.loc[p,entry]-1)/6)
            strength_ratio_values.append(strength_ratio)

            # Calculate the design ratio:
            design_ratio = 1
            design_ratio_values.append(design_ratio)

            #print("Additional UNKNOWN or OTHER calculations for structure entry number: ", p, " of", np.shape(df0)[0], " has been processed for component: ", entry,'at', datetime.datetime.now() - startTime,"------------")

    # Append columns of values to df0:
    df0[entry + '_' + 'des_life_adjustment'] = design_life_adjustment_values
    df0[entry + '_' + 'des_life_adjusted'] = design_life_adjusted_values
    df0[entry + '_' + 'strength_ratio'] = strength_ratio_values
    df0[entry + '_' + 'design_ratio'] = design_ratio_values
    df0[entry + '_' + 'cov'] = cov_values

print("Time to assemble df0 with design life adjusted, cov, and p_f at forecast windspeed values:",datetime.datetime.now() - startTime,"------------")

#---------------------------------------------------------------#
# Begin probability of failure (p_f) calculations at windspeeds #
#---------------------------------------------------------------#
# Calculate p_f values at 1 mph increments from 0 to 120 mph:
startTime = datetime.datetime.now()

# Pronto themes that always use mu_steel
df0['mean_ANCHOR'] = df0['ANCHOR_CD_strength_ratio'] * df0['ANCHOR_CD_design_ratio'] * parameters.mu_steel
df0['stddev_ANCHOR'] = df0['ANCHOR_CD_strength_ratio'] * df0['ANCHOR_CD_design_ratio'] * df0['ANCHOR_CD_cov'] * parameters.mu_steel
df0['mean_GUY'] = df0['GUY_CD_strength_ratio'] * df0['GUY_CD_design_ratio'] * parameters.mu_steel
df0['stddev_GUY'] = df0['GUY_CD_strength_ratio'] * df0['GUY_CD_design_ratio'] * df0['GUY_CD_cov'] * parameters.mu_steel
df0['mean_CONDUCTOR'] = df0['CONDUCTOR_CD_strength_ratio'] * df0['CONDUCTOR_CD_design_ratio'] * parameters.mu_steel
df0['stddev_CONDUCTOR'] = df0['CONDUCTOR_CD_strength_ratio'] * df0['CONDUCTOR_CD_design_ratio'] * df0['CONDUCTOR_CD_cov'] * parameters.mu_steel
df0['mean_OGW'] = df0['OGW_CD_strength_ratio'] * df0['OGW_CD_design_ratio'] * parameters.mu_steel
df0['stddev_OGW'] = df0['OGW_CD_strength_ratio'] * df0['OGW_CD_design_ratio'] * df0['OGW_CD_cov'] * parameters.mu_steel
df0['mean_HI'] = df0['HARDWARE_INSUL_CD_strength_ratio'] * df0['HARDWARE_INSUL_CD_design_ratio'] * parameters.mu_steel
df0['stddev_HI'] = df0['HARDWARE_INSUL_CD_strength_ratio'] * df0['HARDWARE_INSUL_CD_design_ratio'] * df0['HARDWARE_INSUL_CD_cov'] * parameters.mu_steel

# Pronto themes for which whether to use mu_steel or mu_wood depends on the material flag

df0['mu'] = [parameters.mu_steel if x == 'STEEL' else parameters.mu_wood for x in df0['MATERIAL_FLAG']]

df0['mean_FOUNDATION'] =  df0['FOUNDATION_CD_strength_ratio'] * df0['FOUNDATION_CD_design_ratio'] * df0['mu']
df0['stddev_FOUNDATION'] = df0['FOUNDATION_CD_strength_ratio'] * df0['FOUNDATION_CD_design_ratio'] * df0['FOUNDATION_CD_cov'] * df0['mu']
df0['mean_STUB_SPLICE'] = df0['STUB_SPLICE_CD_strength_ratio'] * df0['STUB_SPLICE_CD_design_ratio'] * df0['mu']
df0['stddev_STUB_SPLICE'] = df0['STUB_SPLICE_CD_strength_ratio'] * df0['STUB_SPLICE_CD_design_ratio'] * \
                     df0['STUB_SPLICE_CD_cov'] * df0['mu']
df0['mean_STRUCT_ATTACH'] = df0['STRUCT_ATTACH_CD_strength_ratio'] * df0['STRUCT_ATTACH_CD_design_ratio'] * df0['mu']
df0['stddev_STRUCT_ATTACH'] = df0['STRUCT_ATTACH_CD_strength_ratio'] * df0['STRUCT_ATTACH_CD_design_ratio'] * \
                       df0['STRUCT_ATTACH_CD_cov'] * df0['mu']

print("Time for cov calculations",datetime.datetime.now() - startTime)
df_times.append((datetime.datetime.now() - startTime).total_seconds())

startTime = datetime.datetime.now()
for wspeed in range(0,121):
    col_label = "_" + str(wspeed) + "_mph"
    df0[col_label] = ComputeProbabilityFailureLogNorm(wspeed, df0['mean_ANCHOR'].values, df0['stddev_ANCHOR'].values, df0['mean_GUY'].values, df0['stddev_GUY'].values, df0['mean_FOUNDATION'].values, df0['stddev_FOUNDATION'].values, df0['mean_STUB_SPLICE'].values, df0['stddev_STUB_SPLICE'].values, df0['mean_STRUCT_ATTACH'].values, df0['stddev_STRUCT_ATTACH'].values, df0['mean_CONDUCTOR'].values, df0['stddev_CONDUCTOR'].values, df0['mean_OGW'].values, df0['stddev_OGW'].values, df0['mean_HI'].values, df0['stddev_HI'].values)

print("Time for wind speed calculations",datetime.datetime.now() - startTime)
df_times.append((datetime.datetime.now() - startTime).total_seconds())

# Add date and time at which script was run
df0['DATETIME'] = datetime.datetime.now()


if (writeCSV):
    # Output data calculations to csv:
    print("Writing to csv")
    drop_list = ['WEAR_FATIGUE_RED_FAC', 'AGRICULTURE', 'WETLAND_TYPE',
                 'CORROSION_ZONE', 'INSTALLED_YEAR', 'MATERIAL_FLAG', 'ANCHOR_CD', 'GUY_CD', 'STRUCTURE_CD',
                 'FOUNDATION_CD', 'CROSSARMS_CD', 'FRAME_ATTACH_CD', 'STRUCT_ATTACH_CD', 'STUB_SPLICE_CD',
                 'CONDUCTOR_CD', 'OGW_CD', 'HARDWARE_INSUL_CD', 'SPLICES','TLINE_MILES','OUTAGE_DESIGNLIFE_MOD']
    df0.drop(columns= drop_list).to_csv('df0_calculations.csv')


if (writeDB):
    print("Writing to database")
    try:
        params = urllib.parse.quote_plus(driver = '{SQL Server}', server = config.ExpoServer, database = config.ExpoDatabase, trusted_connection = 'yes')
        conn = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        df0 = df0.replace([np.inf, -np.inf], np.nan)
        df0.drop(columns= drop_list).to_sql('test_Reliability',conn , if_exists = 'replace',index= False,
        dtype={'SAP_EQUIP_ID' : INTEGER,
        'ETGIS_ID' : VARCHAR(50) ,
        'STRUCTURE_NO' : VARCHAR(50) ,
        'SAP_FUNC_LOC_NO' : VARCHAR(50) ,
        'HOST_TLINE_NM' : VARCHAR(100),
        'WSIP_SCOPE_IND' : VARCHAR(1),
        'ANCHOR_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'ANCHOR_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'ANCHOR_CD_strength_ratio' : DECIMAL(8, 6) ,
        'ANCHOR_CD_design_ratio' : DECIMAL(8, 6) ,
        'ANCHOR_CD_cov' : DECIMAL(18, 15),
        'GUY_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'GUY_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'GUY_CD_strength_ratio' : DECIMAL(8, 6) ,
        'GUY_CD_design_ratio' : DECIMAL(8, 6) ,
        'GUY_CD_cov' : DECIMAL(18, 15),
        'FOUNDATION_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'FOUNDATION_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'FOUNDATION_CD_strength_ratio' : DECIMAL(8, 6) ,
        'FOUNDATION_CD_design_ratio' : DECIMAL(8, 6) ,
        'FOUNDATION_CD_cov' : DECIMAL(18, 15),
        'STUB_SPLICE_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'STUB_SPLICE_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'STUB_SPLICE_CD_strength_ratio' : DECIMAL(8, 6) ,
        'STUB_SPLICE_CD_design_ratio' : DECIMAL(8, 6) ,
        'STUB_SPLICE_CD_cov' : DECIMAL(18, 15),
        'STRUCT_ATTACH_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'STRUCT_ATTACH_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'STRUCT_ATTACH_CD_strength_ratio' : DECIMAL(8, 6) ,
        'STRUCT_ATTACH_CD_design_ratio' : DECIMAL(8, 6) ,
        'STRUCT_ATTACH_CD_cov' : DECIMAL(18, 15),
        'CONDUCTOR_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'CONDUCTOR_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'CONDUCTOR_CD_strength_ratio' : DECIMAL(8, 6) ,
        'CONDUCTOR_CD_design_ratio' : DECIMAL(8, 6) ,
        'CONDUCTOR_CD_cov' : DECIMAL(18, 15),
        'OGW_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'OGW_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'OGW_CD_strength_ratio' : DECIMAL(8, 6) ,
        'OGW_CD_design_ratio' : DECIMAL(8, 6) ,
        'OGW_CD_cov' : DECIMAL(18, 15),
        'HARDWARE_INSUL_CD_des_life_adjustment' : DECIMAL(18, 15) ,
        'HARDWARE_INSUL_CD_des_life_adjusted' : DECIMAL(18, 15) ,
        'HARDWARE_INSUL_CD_strength_ratio' : DECIMAL(8, 6) ,
        'HARDWARE_INSUL_CD_design_ratio' : DECIMAL(8, 6) ,
        'HARDWARE_INSUL_CD_cov' : DECIMAL(18, 15),
        'mean_ANCHOR' : DECIMAL(6, 3) ,
        'stddev_ANCHOR' : DECIMAL(20, 15),
        'mean_GUY' : DECIMAL(6, 3) ,
        'stddev_GUY' : DECIMAL(20, 15),
        'mean_CONDUCTOR' : DECIMAL(6, 3) ,
        'stddev_CONDUCTOR' : DECIMAL(20, 15),
        'mean_OGW' : DECIMAL(6, 3) ,
        'stddev_OGW' : DECIMAL(20, 15),
        'mean_HI' : DECIMAL(6, 3) ,
        'stddev_HI' : DECIMAL(20, 15),
        'mu' : DECIMAL(6, 3) ,
        'mean_FOUNDATION' : DECIMAL(6, 3) ,
        'stddev_FOUNDATION' : DECIMAL(20, 15),
        'mean_STUB_SPLICE' : DECIMAL(6, 3) ,
        'stddev_STUB_SPLICE' : DECIMAL(20, 15),
        'mean_STRUCT_ATTACH' : DECIMAL(6, 3) ,
        'stddev_STRUCT_ATTACH' : DECIMAL(20, 15),
        '_0_mph' : DECIMAL(16, 15),
        '_1_mph' : DECIMAL(16, 15),
        '_2_mph' : DECIMAL(16, 15),
        '_3_mph' : DECIMAL(16, 15),
        '_4_mph' : DECIMAL(16, 15),
        '_5_mph' : DECIMAL(16, 15),
        '_6_mph' : DECIMAL(16, 15),
        '_7_mph' : DECIMAL(16, 15),
        '_8_mph' : DECIMAL(16, 15),
        '_9_mph' : DECIMAL(16, 15),
        '_10_mph' : DECIMAL(16, 15),
        '_11_mph' : DECIMAL(16, 15),
        '_12_mph' : DECIMAL(16, 15),
        '_13_mph' : DECIMAL(16, 15),
        '_14_mph' : DECIMAL(16, 15),
        '_15_mph' : DECIMAL(16, 15),
        '_16_mph' : DECIMAL(16, 15),
        '_17_mph' : DECIMAL(16, 15),
        '_18_mph' : DECIMAL(16, 15),
        '_19_mph' : DECIMAL(16, 15),
        '_20_mph' : DECIMAL(16, 15),
        '_21_mph' : DECIMAL(16, 15),
        '_22_mph' : DECIMAL(16, 15),
        '_23_mph' : DECIMAL(16, 15),
        '_24_mph' : DECIMAL(16, 15),
        '_25_mph' : DECIMAL(16, 15),
        '_26_mph' : DECIMAL(16, 15),
        '_27_mph' : DECIMAL(16, 15),
        '_28_mph' : DECIMAL(16, 15),
        '_29_mph' : DECIMAL(16, 15),
        '_30_mph' : DECIMAL(16, 15),
        '_31_mph' : DECIMAL(16, 15),
        '_32_mph' : DECIMAL(16, 15),
        '_33_mph' : DECIMAL(16, 15),
        '_34_mph' : DECIMAL(16, 15),
        '_35_mph' : DECIMAL(16, 15),
        '_36_mph' : DECIMAL(16, 15),
        '_37_mph' : DECIMAL(16, 15),
        '_38_mph' : DECIMAL(16, 15),
        '_39_mph' : DECIMAL(16, 15),
        '_40_mph' : DECIMAL(16, 15),
        '_41_mph' : DECIMAL(16, 15),
        '_42_mph' : DECIMAL(16, 15),
        '_43_mph' : DECIMAL(16, 15),
        '_44_mph' : DECIMAL(16, 15),
        '_45_mph' : DECIMAL(16, 15),
        '_46_mph' : DECIMAL(16, 15),
        '_47_mph' : DECIMAL(16, 15),
        '_48_mph' : DECIMAL(16, 15),
        '_49_mph' : DECIMAL(16, 15),
        '_50_mph' : DECIMAL(16, 15),
        '_51_mph' : DECIMAL(16, 15),
        '_52_mph' : DECIMAL(16, 15),
        '_53_mph' : DECIMAL(16, 15),
        '_54_mph' : DECIMAL(16, 15),
        '_55_mph' : DECIMAL(16, 15),
        '_56_mph' : DECIMAL(16, 15),
        '_57_mph' : DECIMAL(16, 15),
        '_58_mph' : DECIMAL(16, 15),
        '_59_mph' : DECIMAL(16, 15),
        '_60_mph' : DECIMAL(16, 15),
        '_61_mph' : DECIMAL(16, 15),
        '_62_mph' : DECIMAL(16, 15),
        '_63_mph' : DECIMAL(16, 15),
        '_64_mph' : DECIMAL(16, 15),
        '_65_mph' : DECIMAL(16, 15),
        '_66_mph' : DECIMAL(16, 15),
        '_67_mph' : DECIMAL(16, 15),
        '_68_mph' : DECIMAL(16, 15),
        '_69_mph' : DECIMAL(16, 15),
        '_70_mph' : DECIMAL(16, 15),
        '_71_mph' : DECIMAL(16, 15),
        '_72_mph' : DECIMAL(16, 15),
        '_73_mph' : DECIMAL(16, 15),
        '_74_mph' : DECIMAL(16, 15),
        '_75_mph' : DECIMAL(16, 15),
        '_76_mph' : DECIMAL(16, 15),
        '_77_mph' : DECIMAL(16, 15),
        '_78_mph' : DECIMAL(16, 15),
        '_79_mph' : DECIMAL(16, 15),
        '_80_mph' : DECIMAL(16, 15),
        '_81_mph' : DECIMAL(16, 15),
        '_82_mph' : DECIMAL(16, 15),
        '_83_mph' : DECIMAL(16, 15),
        '_84_mph' : DECIMAL(16, 15),
        '_85_mph' : DECIMAL(16, 15),
        '_86_mph' : DECIMAL(16, 15),
        '_87_mph' : DECIMAL(16, 15),
        '_88_mph' : DECIMAL(16, 15),
        '_89_mph' : DECIMAL(16, 15),
        '_90_mph' : DECIMAL(16, 15),
        '_91_mph' : DECIMAL(16, 15),
        '_92_mph' : DECIMAL(16, 15),
        '_93_mph' : DECIMAL(16, 15),
        '_94_mph' : DECIMAL(16, 15),
        '_95_mph' : DECIMAL(16, 15),
        '_96_mph' : DECIMAL(16, 15),
        '_97_mph' : DECIMAL(16, 15),
        '_98_mph' : DECIMAL(16, 15),
        '_99_mph' : DECIMAL(16, 15),
        '_100_mph' : DECIMAL(16, 15),
        '_101_mph' : DECIMAL(16, 15),
        '_102_mph' : DECIMAL(16, 15),
        '_103_mph' : DECIMAL(16, 15),
        '_104_mph' : DECIMAL(16, 15),
        '_105_mph' : DECIMAL(16, 15),
        '_106_mph' : DECIMAL(16, 15),
        '_107_mph' : DECIMAL(16, 15),
        '_108_mph' : DECIMAL(16, 15),
        '_109_mph' : DECIMAL(16, 15),
        '_110_mph' : DECIMAL(16, 15),
        '_111_mph' : DECIMAL(16, 15),
        '_112_mph' : DECIMAL(16, 15),
        '_113_mph' : DECIMAL(16, 15),
        '_114_mph' : DECIMAL(16, 15),
        '_115_mph' : DECIMAL(16, 15),
        '_116_mph' : DECIMAL(16, 15),
        '_117_mph' : DECIMAL(16, 15),
        '_118_mph' : DECIMAL(16, 15),
        '_119_mph' : DECIMAL(16, 15),
        '_120_mph' : DECIMAL(16, 15),
        'DATETIME' : DATETIME})
    except:
        # (Needs updating) add error handling
        print("Error writing to database")

# Assemble the DataFrame of times for performance checking:
#df_times_performance = pd.DataFrame()
#df_times_performance['LABELS'] = ['ASSEMBLE_USER_INPUT---','READ_IN_DFs---','CALCULATE_cov_VALUES---','CALCULATE_P_F_VALUES---']
#df_times_performance['TIMES'] = df_times
#df_times_performance.to_csv('df_times_performance.txt', sep=',', index=False)

print("Time to complete script:",datetime.datetime.now() - startTime,"------------")

