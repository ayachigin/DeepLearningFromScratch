module Mnist
  ( normalize
  , readTrainDataSets
  , Label
  , Image
  , NormalizedImage
  , DataSet) where

import qualified Data.ByteString as B

type Label = Int

type Image = [Int]

type NormalizedImage = [Double]

type DataSet = (Image, Label)

normalize :: Image -> NormalizedImage
normalize = map ((/255) . fromIntegral)

readTrainDataSets :: IO [DataSet]
readTrainDataSets  = do
  i <- B.readFile "mnist/train-images.idx3-ubyte"
  l <- B.readFile "mnist/train-labels.idx1-ubyte"
  return $ toDataSets (toImgs i) (toLabels l)
  where
    toDataSets = zip
    toImgs = toImgs' . map fromIntegral . B.unpack . B.drop 16
    toImgs' [] = []
    toImgs' ls = (take (28 * 28) ls):
                (toImgs' $ drop (28 * 28) ls)
    toLabels = map fromIntegral . B.unpack . B.drop 8
