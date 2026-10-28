Cash Atlas assets

Bank and PSP icons identify the researched entities. Their trademarks belong to
their respective owners; the dashboard does not imply endorsement. Official-site
asset URLs are recorded in manifest.json. GHL's logo is reproduced from
BusinessWorld company news coverage. Xendit's vector mark comes from the
Simple Icons project: https://github.com/simple-icons/simple-icons

The workspace symbol and interface icons are original geometric line artwork.
Native widget icons use Streamlit's Material Symbols support.

Inter and Manrope fonts are hosted locally under static/fonts. Their SIL Open
Font License notices are included alongside the font files. Sources:
https://github.com/google/fonts/tree/main/ofl/inter
https://github.com/google/fonts/tree/main/ofl/manrope

To refresh assets explicitly:
    uv run python scripts/fetch_brand_assets.py
The dashboard itself never downloads logos, calls research APIs or loads keys.
