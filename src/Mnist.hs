module Mnist
  ( randomNormalizedDatasets
  , randomDatasets
  , Label
  , Image
  , NormalizedImage
  , NormalizedDataSets
  , DataSets) where

import qualified Data.ByteString as B
import System.Random (randomRIO)
import Control.Monad (replicateM)
import Data.Array.Repa hiding (map)
import Data.Array.Repa.Repr.Vector
import Data.Vector.Unboxed.Base (Unbox)

type Label = Int

type Image = [Int]

type NormalizedImage = [Double]

type DataSet = (Image, Label)

type NormalizedDataSet = (NormalizedImage, Label)

type NormalizedDataSets = (Array V DIM1 (Array U DIM2 Double, Label))

type DataSets = (Array V DIM1 (Array U DIM2 Int, Label))

toArray :: (Unbox a) =>
           Int -> [([a], b)] -> Array V DIM1 (Array U DIM2 a, b)
toArray idx ls = v
  where
    v = fromListVector (ix1 idx) $ fmap f ls
    f (i, l) = (fromListUnboxed (ix2 1 (28 * 28)) i, l)

randomNormalizedDatasets :: Int ->
                            IO NormalizedDataSets
randomNormalizedDatasets batsize = do
  datasets <- fmap (normalize) $ choiceIO batsize
  return . toArray batsize $ datasets

randomDatasets :: Int ->
                  IO DataSets
randomDatasets batsize = do
  datasets <- choiceIO batsize
  return . toArray batsize $ datasets


normalize :: [DataSet] -> [NormalizedDataSet]
normalize = map (\(x, y) -> (map ((/255).fromIntegral) x, y))

choice' :: (B.ByteString, B.ByteString) -> [Int] -> [DataSet]
choice' (i, l) idxs = map (\n -> (takeImage n i, takeLabel n l)) idxs
  where
    takeImage n = map fromIntegral . B.unpack . B.take (28 * 28) .
                  B.drop (28 * 28 * n)
    takeLabel n = fromIntegral  . B.head . B.drop n
    --fromList $ fmap (vec !) idxs

choiceIO :: Int -> IO [DataSet]
choiceIO n  = do
  (is, ls) <- readTrainDataSets
  idxs <- randomRIOs is
  return $ choice' (B.drop 16 is, B.drop 8 ls) idxs
  where
    len b = fromIntegral (B.index b 6) * 256 + fromIntegral (B.index b 7)
    range b = (0, len b - 1)
    randomRIOs b = replicateM n (randomRIO $ range b)


readTrainDataSets :: IO (B.ByteString, B.ByteString)
readTrainDataSets  = do
  i <- B.readFile "mnist/train-images.idx3-ubyte"
  l <- B.readFile "mnist/train-labels.idx1-ubyte"
  return $ (i, l)
