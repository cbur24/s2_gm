"""
Geomedian with enhanced s2Cloudless masking

TODO:
- What fusers should I be using on load_with_native_transform?
- How is contiguity and nodata masking working? How is it being resampled?
- Need to load all S2 sensors for long-term cloud probability
- How can we document these functions for new users?
    The plugin examples in odc-stats are  sparse.
    
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
# from odc.stats.plugins import StatsPluginInterface
from odc.stats.plugins._registry import register, StatsPluginInterface
from odc.algo import xr_quantile, geomedian_with_mads
from odc.algo._masking import (
    _xr_fuse,
    _fuse_mean_np,
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
        contiguity_band: str = "nbart_contiguity",
        group_by: str = "solar_day",
        nodata_classes: Optional[Sequence[str]] = ["nodata"], 
        cp_threshold: float = 0.1,
        cloud_filters: Optional[Iterable[Tuple[str, int]]] = [
            ["opening", 2],
            ["dilation", 3],
        ],
        aux_names: Dict[str, str] = None,
        work_chunks: Tuple[int, int] = (1024, 1024),
        output_dtype: str = "float32",
        **kwargs,
    ):
        
        self.bands = bands
        self.mask_band = mask_band
        self.proba_band = proba_band
        self.contiguity_band = contiguity_band
        self.group_by = group_by
        self.resampling = resampling
        self.nodata_classes= nodata_classes
        self.cp_threshold = cp_threshold
        self.cloud_filters = cloud_filters
        self.work_chunks = work_chunks
        # self.aux_names = aux_names
        self.output_dtype = np.dtype(output_dtype)
        self.output_nodata = np.nan
        self._renames = aux_names
        self.aux_names = tuple(
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
            input_bands=bands,
            chunks=work_chunks,
            # mask_band=mask_band,
            # nodata_classes=nodata_classes,
            # # cloud_filters=cloud_filters,
            # aux_names=aux_names,
            **kwargs,
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        return (
            tuple(b for b in self.bands if b != self.contiguity_band) + self.aux_names
        )

    def input_data(self, datasets: Sequence[Dataset], geobox: GeoBox, **kwargs
        ) -> xr.Dataset:
        """
        - Load S2 data, erasing nodata and non-contiguous pixels.
        - Load long-term cloud probability and use it to split
            cloud masking threshold between 'good' and 'bad' pixels.
        - return cloud masked S2 data.
        """

        def masking_nodata(xx: xr.Dataset) -> xr.Dataset:
            """
            Only interested in erasing the nodata and non-contiguous
            pixels, as cloud masking is handled later.
            """
            if self.mask_band not in xx.data_vars:
                return xx
            
            # Erase Data Pixels for which mask == nodata
            mask = xx[self.mask_band]
            bad = enum_to_bool(mask, self.nodata_classes)
            
            # Apply the contiguity flag
            if self.contiguity_band is not None:
                non_contiguent = xx.get(self.contiguity_band, 1) == 0
                bad = bad | non_contiguent
    
            if self.contiguity_band is not None:
                xx = xx.drop_vars([self.mask_band] + [self.contiguity_band])
            else:
                xx = xx.drop_vars([self.mask_band])
            
            xx = erase_bad(xx, bad)
            
            return xx
        
        #Load (annual) Sentinel 2 data
        s2 = load_with_native_transform(
            dss=datasets,
            geobox=geobox,
            native_transform=lambda x: masking_nodata(x),
            bands=tuple(self.bands) + (self.proba_band,),
            groupby=self.group_by,
            # fuser=self.fuser, #WHAT TO DO HERE?
            chunks=dict(y=self.work_chunks[0], x=self.work_chunks[0]),
            resampling=self.resampling,
        )

        #-------Cloud probability---------------------------------

        """
        Load a long timeseries of S2Cloudless probability,
        calculate 10th percentiles, generate cloud mask,
        apply morphological filters.

        cp_threshold = The threshold value for identifying
                regions that are commonly missclassified
                as cloud. If the long-term 10th percentile
                probability is larger than this values its
                considered 'bad'. 

        Logic:
        1. Take the 10th percentile of long-term cloud probabilities (CP)
        2. Where 10th percentile CP > cp_threshold, add 0.4 to the long-term percentiles,
           this is the new cloud-probability threshold for those problem regions.
        3. Clip the maximum threshold to 0.90 (highest threshold is 90 %)
        """
        # This doesn't fit the odc-stats paradigm so hardcode loading of
        #  long-term cloud probabilities with dc.load 
        dc = datacube.Datacube(app="load cloud probs")

        #load long-term cloud-probabilities
        # just testing with one sensor at first
        cloud_probs = dc.load(
            product="ga_s2bm_ard_3", #need to do this with all sensors
            like=s2.geobox, #match spatial extents
            time=('2020', '2025'), #long-term
            measurements=['oa_s2cloudless_prob'],
            dask_chunks=dict(x=self.work_chunks[0], y=self.work_chunks[0], time=-1),
            resampling=self.resampling,
            group_by='solar_day'
        )
    
        #Calculate long-term 10th percentile
        prob_quantile = xr_quantile(cloud_probs[['oa_s2cloudless_prob']], quantiles=[0.1], nodata=np.nan)
        prob_quantile = prob_quantile['oa_s2cloudless_prob'].sel(quantile=0.1)

        # this should work but acts weird when applying morphological filter
        # updated_cloud_mask = xr.where(prob_quantile > cp_threshold, #where 10th % cp is above 0.1:
        #     s2['oa_s2cloudless_prob'] > (prob_quantile+0.4).clip(0, 0.90), # threshold probability by 0.4 + cp_10th_%
        #     s2['oa_s2cloudless_prob'] > 0.4, #otherwise just threshold using 0.4
        #                      ).drop_vars('quantile')

        # cloud mask for regions repeatedly misclassified as cloud 
        bad_regions = xr.where(prob_quantile>self.cp_threshold, True, False) 
        bad_regions_proba = s2['oa_s2cloudless_prob'].where(bad_regions)
        bad_regions_proba_mask = xr.where(bad_regions_proba>=(prob_quantile+0.4).clip(0, 0.90), True, False)
        
        # cloud mask for regions NOT repeatedly misclassified as cloud, threshold with 0.4
        good_regions = xr.where(prob_quantile<=self.cp_threshold, True, False)
        good_regions_proba = s2['oa_s2cloudless_prob'].where(good_regions)
        good_regions_proba_mask = xr.where(good_regions_proba>0.4, True, False)
        
        ## Combine cloud masks
        updated_cloud_mask = np.logical_or(
            bad_regions_proba_mask, good_regions_proba_mask
                    )
        
        # apply morphological filters
        updated_cloud_mask = mask_cleanup(updated_cloud_mask, mask_filters=self.cloud_filters)

        #tidy up and apply the cloud mask to the data
        s2 = s2.drop_vars('oa_s2cloudless_prob')
        s2 = s2.where(~updated_cloud_mask).drop_vars('quantile')

        return s2

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        """
        Run the geomedian on the masked S2 data.
        
        """
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