"""
Sentinel-2 Geomedian with enhanced s2Cloudless masking

TODO:
- Should I be doing something with fusers?
- Consider how nodata is handled, explicitly add to outputs?

"""

from functools import partial
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import datacube
import numpy as np
import pandas as pd
import xarray as xr
from datacube.model import Dataset
from datacube.utils import masking
from odc.geo.xr import assign_crs
from odc.geo.geobox import GeoBox
from odc.algo.io import load_with_native_transform
from odc.stats.plugins._registry import register, StatsPluginInterface
from odc.algo import xr_quantile, geomedian_with_mads
from odc.algo._masking import (
    erase_bad,
    enum_to_bool,
    mask_cleanup,
)

class GMS2AUS(StatsPluginInterface):
    NAME = "GMS2AUS"
    SHORT_NAME = NAME
    VERSION = "1.0.0"
    PRODUCT_FAMILY = "geomedian"

    def __init__(
        self,
        resampling: str = "cubic",
        bands: Optional[Sequence[str]] = ["nbart_red", "nbart_green", "nbart_blue"],
        mask_band: str = "oa_s2cloudless_mask",
        proba_band: str = "oa_s2cloudless_prob",
        contiguity_band: str = "oa_nbart_contiguity",
        s2cloudless_enchance: bool = True,
        group_by: str = "solar_day",
        nodata_classes: Optional[Sequence[str]] = ["nodata"], 
        cp_threshold: float = 0.1,
        year: str = None,
        cloud_filters: Optional[Iterable[Tuple[str, int]]] = [
            ["opening", 6],
            ["dilation", 10],
        ],
        aux_names: Dict[str, str] = None,
        work_chunks: Tuple[int, int] = (400,400),
        output_dtype: str = "float32",
        **kwargs,
    ):
        aux_names = (
            {"smad": "smad", "emad": "emad", "bcmad": "bcmad", "count": "count"}
            if aux_names is None
            else aux_names
        )
        self.bands = bands
        self.mask_band = mask_band
        self.proba_band = proba_band
        self.contiguity_band = contiguity_band
        self.group_by = group_by
        self.resampling = resampling
        self.s2cloudless_enchance = s2cloudless_enchance
        self.nodata_classes= nodata_classes
        self.cp_threshold = cp_threshold
        self.year = year
        self.cloud_filters = cloud_filters
        self.work_chunks = work_chunks
        self.aux_names = aux_names
        self.output_dtype = np.dtype(output_dtype)
        self.output_nodata = np.nan
        self._renames = aux_names
        self.aux_bands = tuple(
            self._renames.get(k, k)
            for k in (
                "smad",
                "emad",
                "bcmad",
                "count",
            )
        )

        if bands is None:
            bands = (
                "nbart_red", "nbart_green", "nbart_blue"
            )
            
            if rgb_bands is None:
                rgb_bands = ("nbart_red", "nbart_green", "nbart_blue")

        super().__init__(
            input_bands=tuple(bands)+(mask_band,)+(proba_band,)+(contiguity_band,), 
            **kwargs
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        return (
            tuple(b for b in self.bands if b != self.contiguity_band) + self.aux_bands
        )

    def native_transform(self, xx: xr.Dataset) -> xr.Dataset:
        """
        The first step in this transform is similar to standard
        GM plugin in that it erases nodata and non-contiguous pixels.
        Cloud masking comes later in the reduce step
        
        """

        # step 1-----------------
        if self.mask_band not in xx.data_vars:
            return xx

        # Erase Data Pixels for which mask == nodata
        mask = xx[self.mask_band]
        bad = enum_to_bool(mask, self.nodata_classes)
        
         # Apply the contiguity flag
        if self.contiguity_band is not None:
            non_contiguent = xx.get(self.contiguity_band, 1) == 0
            bad = bad | non_contiguent

        if not self.s2cloudless_enchance:
            # in this case, just use the standard s2cloudless
            # cloud mask, and we'll skip using the
            # probability layer
            cloud_mask = enum_to_bool(
                    mask=xx[self.mask_band], categories=["cloud"]
                )
            bad = cloud_mask | bad

        # drop masking bands
        if self.contiguity_band is not None:
            xx = xx.drop_vars([self.mask_band] + [self.contiguity_band])
        else:
            xx = xx.drop_vars([self.mask_band])

        # apply the masks
        xx = erase_bad(xx, bad)

        return xx
         

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        """
        If s2cloudless_enchance == True, the we use the
        long-term cloud-probabilities to find regions that are
        persistently misclassified as cloud.

        Logic:
        1. Take the 10th percentile of long-term cloud probabilities (CP)
        2. Where 10th percentile CP > cp_threshold, add 0.4 to the long-term percentiles,
           this is the new cloud-probability threshold for those problem regions.
        3. Clip the maximum threshold to 0.90 (highest threshold is 90 %)
        
        Then we apply morphological filters to the enhanced 
        cloud mask.

        Then run the geomedian on the masked S2 data, for a selected year. We need
        to select the year because for compting the enhanced cloud probabilities
        percentiles we've cached (with odc.-stats.save-tasks) a longer time series.

        """
        if self.s2cloudless_enchance:
            # step 1. ---Use the cloud probability to identify persistently mis-classified regions--.
            #  Compute the 10th percentile of long-term cloud probabilities.
            prob_quantile = xr_quantile(xx[[self.proba_band]],quantiles=[0.1], nodata=np.nan)
            prob_quantile = prob_quantile[self.proba_band].sel(quantile=0.1)

            # Select a single year of data as we only needed multiple years
            # for the cloud probabilities quantiles.
            xx = xx.sel(time=self.year)
    
            # step 2+3. cloud mask for regions repeatedly misclassified as cloud.
            bad_regions = xr.where(prob_quantile>self.cp_threshold, True, False) 
            bad_regions_proba = xx[self.proba_band].where(bad_regions)
            bad_regions_proba_mask = xr.where(bad_regions_proba>=(prob_quantile+0.4).clip(0, 0.90), True, False)
            
            # cloud mask for regions NOT repeatedly misclassified as cloud, threshold with default 0.4
            good_regions_proba = xx[self.proba_band].where(~bad_regions)
            good_regions_proba_mask = xr.where(good_regions_proba>0.4, True, False)
            
            ## Combine cloud masks
            updated_cloud_mask = np.logical_or(
                bad_regions_proba_mask, good_regions_proba_mask
                        )
            
            # -------apply morphological filters--------------
            updated_cloud_mask = mask_cleanup(updated_cloud_mask, mask_filters=self.cloud_filters)
    
            # apply the cloud mask to the data
            xx = erase_bad(xx, updated_cloud_mask)
    
        # drop probability layer
        xx = xx.drop_vars(self.proba_band)

        if not self.s2cloudless_enchance:
            # ensure we're grabbing just the GM year in the case
            # where we didn't change the cloud mask probabilities.
            xx = xx.sel(time=self.year) 

        #-----run the GM, config for gm below---
        scale = 1 / 10_000
        cfg = {
            "maxiters": 1000,
            "num_threads": 1,
            "scale": scale,
            "offset": -1 * scale,
            "reshape_strategy": "mem",
            "out_chunks": (-1, -1, -1),
            "work_chunks": self.work_chunks,
            "compute_count": True,
            "compute_mads": False,
        }
        
        gm = geomedian_with_mads(xx, **cfg)

        return gm

register("s2_gm_tools.GMS2AUS", GMS2AUS)
