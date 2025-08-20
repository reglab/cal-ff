# Cal-FF Github Repository

This repository contains much of the code used to collate and curate Cal-FF.

Cal-FF is a comprehensive dataset of Concentrated Animal Feeding Operations (CAFOs) in California, compiled using satellite imagery, computer vision, and human validation. It provides an improved and near-complete census of CAFOs, addressing gaps in administrative records.

This dataset has redacted some publicly-available information, including facility addresses, in order to protect privacy. Because these are highly regulated facilities and addresses are a key method of performing record linkage and using this data in research, we will share this data for research purposes upon request.

Note: the animal type data described below is not available in the geoparquet, but it is in the geojson file.

## Dataset Details

### Dataset Description

Cal-FF includes facility locations, ownership records, satellite images, and detailed annotations. It was compiled to improve upon incomplete or inaccurate state administrative records.

- **Curated by:** Varun Magesh, Nic Rothbacher, Saskia Comess, Erin Maneri, Kit Rodolfa, Sara Tartof, Joan Casey, Keeve Nachman, Daniel E. Ho
- **Funded by:** Anonymous Fund at the Greater Kansas City Community Foundation, Schmidt Futures, Google Cloud
- **License:** the Cal-FF Dataset © 2025 by Varun Magesh, Nic Rothbacher, Saskia Comess, Erin Maneri, Kit Rodolfa, Sara Tartof, Joan Casey, Keeve Nachman, Daniel E. Ho is marked with [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). If used in an academic work, please attribute to:

Magesh, Varun, Nic Rothbacher, Saskia Comess, Erin Maneri, Kit Rodolfa, Sara Tartof, Joan Casey, Keeve Nachman and Daniel E. Ho. Cal-FF: A Comprehensive Dataset of Factory Farms in California Compiled Using Computer Vision and Human Validation. Unpublished manuscript, 2025. Available at [https://huggingface.co/datasets/reglab/cal-ff](https://huggingface.co/datasets/reglab/cal-ff).

### Dataset Sources

- **Repository:** [https://huggingface.co/datasets/reglab/cal-ff](https://huggingface.co/datasets/reglab/cal-ff)
- **Paper:** Submitted for review.

## Uses

### Direct Use

- Improved CAFO location tracking
- Analysis of regulatory gaps
- Research on environmental and public health impacts of factory farming

## Dataset Contents

- **Facility Footprints**: Geospatial data about facility building footprint labels
- **Regulatory Data**: CAFO permits associated with facilities
- **Land use data**: Parcel information
- **Annotations**: Footprints, animal types, construction & destruction dates

## Dataset Creation

### Curation Rationale

To correct and expand upon existing CAFO datasets used in regulation and public health research.

### Source Data

#### Data Collection and Processing

- CV model used on satellite imagery to detect CAFOs
- Manual review by trained annotators
- Cross-referenced with state permit data and parcel maps

#### Who are the source data producers?

- Satellite imagery providers (e.g., Google)
- Public data sources (ReGrid parcel data; CWIQS CA permit data)
- Environmental Working Group (provided initial locations)
- CloudFactory labeling team
- Undergraduate research assistants

### Annotations

#### Annotation process

- Human-in-the-loop validation and correction of model predictions
- Labeling of animal types and facility boundaries

#### Who are the annotators?

- Undergraduate researchers
- CloudFactory contractors
- Authors

#### Personal and Sensitive Information

The dataset does not include private PII. Facility locations are publicly visible and cross-referenced with public parcel data.


## Acknowledgements

The team would like to thank an Anonymous Fund at the Greater Kansas City Community Foundation, Schmidt Futures and Windward Fund for supporting this work. We thank Google Earth Engine for access to their analysis platform, the Environmental Working Group for providing data and insights on identifying CAFOs, and CloudFactory for contributing data annotation support. We also thank Ben Chugg for data analysis & model engineering assistance, and Vincent La and Dana Stokes for contributing software and data engineering support. Finally, we thank Brandon Anderson and Elena Eneva for their guidance, Christine Tsang for management support, and Arun Frey, Helena Lyng-Olsen, Yumna Naqvi, and Deborah Sivas for helpful conversation and suggestions.
