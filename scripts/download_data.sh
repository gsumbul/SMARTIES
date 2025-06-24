# Pretraining data
wget https://stacks.stanford.edu/file/druid:vg497cb6002/README.md -P data/fMoW-S2/
wget https://stacks.stanford.edu/file/druid:vg497cb6002/fmow-sentinel.tar.gz -P data/fMoW-S2/
wget https://stacks.stanford.edu/file/druid:vg497cb6002/test_gt.csv -P data/fMoW-S2/
wget https://stacks.stanford.edu/file/druid:vg497cb6002/train.csv -P data/fMoW-S2/
wget https://stacks.stanford.edu/file/druid:vg497cb6002/val.csv -P data/fMoW-S2/
tar -xvf data/fMoW-S2/fmow-sentinel.tar.gz -C data/fMoW-S2
rm -f data/fMoW-S2/fmow-sentinel.tar.gz

aws s3 sync --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-full/ data/fMoW/
tar -xvf data/fMoW/groundtruth.tar.bz2 -C data/fMoW/
rm data/fMoW/groundtruth.tar.bz2

aws s3 sync --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/ data/fMoW-RGB

wget -r ftp://m1660427.001:m1660427.001@dataserv.ub.tum.de/ -P data/SSL4EO-S12/
tar -xvf data/SSL4EO-S12/ssl4eo-s12/s1.tar.gz -C data/SSL4EO-S12/ssl4eo-s12/
rm data/SSL4EO-S12/ssl4eo-s12/s1.tar.gz
tar -xvf data/SSL4EO-S12/ssl4eo-s12/s2_l1c.tar.gz -C data/SSL4EO-S12/ssl4eo-s12/
rm data/SSL4EO-S12/ssl4eo-s12/s2_l1c.tar.gz
tar -xvf data/SSL4EO-S12/ssl4eo-s12/s2_l2a.tar.gz -C data/SSL4EO-S12/ssl4eo-s12/
rm data/SSL4EO-S12/ssl4eo-s12/s2_l2a.tar.gz

# Evaluation data
wget https://zenodo.org/records/7711810/files/EuroSAT_MS.zip -P data/EuroSAT/
unzip data/EuroSAT/EuroSAT_MS.zip -d data/EuroSAT/
rm data/EuroSAT/EuroSAT_MS.zip

wget https://github.com/CAPTAIN-WHU/BED4RS/raw/main/datasets/WHU-RS19.zip -P data/WHU-RS19/
unzip data/WHU-RS19/WHU-RS19.zip -d data/WHU-RS19/
mv data/WHU-RS19/WHU-RS19/* data/WHU-RS19/
rm -r data/WHU-RS19/WHU-RS19/
rm data/WHU-RS19/WHU-RS19.zip

wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip -P data/UCMerced/
unzip data/UCMerced/UCMerced_LandUse.zip -d data/UCMerced/
mv data/UCMerced/UCMerced_LandUse/* data/UCMerced/
rm -r data/UCMerced/UCMerced_LandUse
rm data/UCMerced/UCMerced_LandUse.zip

wget https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz -P data/BigEarthNet-v1/
tar -xvf data/BigEarthNet-v1/BigEarthNet-S2-v1.0.tar.gz -C data/BigEarthNet-v1/
mv data/BigEarthNet-v1/BigEarthNet-v1.0 data/BigEarthNet-v1/S2
rm data/BigEarthNet-v1/BigEarthNet-S2-v1.0.tar.gz

wget https://zenodo.org/records/12687186/files/BigEarthNet-S1-v1.0.tar.gz -P data/BigEarthNet-v1/
tar -xvf data/BigEarthNet-v1/BigEarthNet-S1-v1.0.tar.gz -C data/BigEarthNet-v1/
mv data/BigEarthNet-v1/BigEarthNet-S1-v1.0 data/BigEarthNet-v1/S1
rm data/BigEarthNet-v1/BigEarthNet-S1-v1.0.tar.gz

wget https://huggingface.co/datasets/antofuller/CROMA_benchmarks/resolve/main/DFC_preprocessed.pt -P data/DFC2020/

wget https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/resolve/main/hls_burn_scars.tar.gz -P data/HLSBurnScars/
tar -xvf data/HLSBurnScars/hls_burn_scars.tar.gz -P data/HLSBurnScars/

aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz data/SpaceNet7/
aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz data/SpaceNet7/
aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz data/SpaceNet7/

tar -xvf data/SpaceNet7/SN7_buildings_train.tar.gz -C data/SpaceNet7/
tar -xvf data/SpaceNet7/SN7_buildings_train_csvs.tar.gz -C data/SpaceNet7/
tar -xvf data/SpaceNet7/SN7_buildings_test_public.tar.gz -C data/SpaceNet7/

rm data/SpaceNet7/SN7_buildings_train.tar.gz
rm data/SpaceNet7/SN7_buildings_train_csvs.tar.gz
rm data/SpaceNet7/SN7_buildings_test_public.tar.gz