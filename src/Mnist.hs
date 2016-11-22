module Mnist
  ( randomNormalizedDataset
  , randomDataSet
  , Label
  , Image
  , NormalizedImage
  , NormalizedDataSet
  , DataSet) where

import Prelude hiding (map)
import qualified Data.ByteString as B
import System.Random (randomRIO)
import Control.Monad (replicateM)
import Data.Array.Repa hiding ((++))

type Label = Int

type Image = Array U DIM2 Int

type NormalizedImage = Array U DIM2 Double

type DataSet = (Array U DIM2 Int, Array U DIM2 Double)

type NormalizedDataSet = (Array U DIM2 Double, Array U DIM2 Double)

randomNormalizedDataset :: Int ->
                            IO NormalizedDataSet
randomNormalizedDataset batsize = fmap (normalize) $ randomDataSet batsize

randomDataSet :: Int -> IO DataSet
randomDataSet n  = do
  (is, ls) <- readTrainDataSets
  idxs <- randomRIOs is
  return $ choice' (B.drop 16 is, B.drop 8 ls) idxs
  where
    len b = fromIntegral (B.index b 6) * 256 + fromIntegral (B.index b 7)
    range b = (0, len b - 1)
    randomRIOs b = replicateM n (randomRIO $ range b)

normalize :: DataSet -> NormalizedDataSet
normalize (i, l) = (computeS $ map ((/255).fromIntegral) i, l)

choice' :: (B.ByteString, B.ByteString) -> [Int] -> DataSet
choice' (i, l) idxs = (imgs, labels)
--  map (\n -> (takeImage n i, takeLabel n l)) idxs
  where
    len = length idxs
    imgs = fromListUnboxed (ix2 len (28*28)) $
           concat [takeImage n i | n <-idxs]
    labels = fromListUnboxed (ix2 len 10) $
             concat [takeLabel n l | n <- idxs]
    takeImage n = fmap fromIntegral . B.unpack . B.take (28 * 28) .
                  B.drop (28 * 28 * n)
    takeLabel n = shift bits . fromIntegral  . B.head . B.drop n
    shift [] _ = error "Shift empty list"
    shift ls 0 = ls
    shift (x:xs) n = shift (xs ++ [x]) (n-1)
    bits = [(1.0::Double), 0, 0, 0, 0, 0, 0, 0, 0, 0]

readTrainDataSets :: IO (B.ByteString, B.ByteString)
readTrainDataSets  = do
  i <- B.readFile "mnist/train-images.idx3-ubyte"
  l <- B.readFile "mnist/train-labels.idx1-ubyte"
  return $ (i, l)
