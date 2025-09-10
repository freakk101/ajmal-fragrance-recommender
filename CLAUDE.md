# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a data repository containing fragrance information scraped from the Ajmal perfume website. The repository currently contains:

- `chrome_ajmal_fragrance_data_20250909_202423.csv`: CSV file containing fragrance product data including:
  - Product URLs and images
  - Product names and prices
  - Fragrance notes (top, heart/middle, base notes)
  - Product variants and descriptions

## Data Structure

The CSV contains fragrance data with the following key columns:
- `media href`: Product page URLs
- `image src`: Product image URLs  
- `product-card__title`: Product names
- `tw-text-[rgba(var(--color-foreground))]`: Current prices
- `tw-text-[#a38d6f]`: Original prices
- `Chrome_Top_Notes`: Top fragrance notes
- `Chrome_Heart_Notes`: Heart/middle fragrance notes
- `Chrome_Base_Notes`: Base fragrance notes

## Development Context

This appears to be a data collection project for building a fragrance recommendation system. The repository is currently in a data-only state with no application code, build tools, or testing infrastructure present.

Future development might involve:
- Data processing and cleaning scripts
- Recommendation algorithm implementation
- Web application or API development
- Database integration