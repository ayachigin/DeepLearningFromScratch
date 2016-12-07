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
import qualified Data.ByteString.Lazy as BL
import System.Random (randomRIO)
import Control.Monad (replicateM)
import Data.Array.Repa hiding ((++))
import System.Directory
import Control.Lens
import Network.Wreq (get, responseBody)
import Codec.Compression.GZip (decompress)

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
  downloadDatasetsIfMissing
  i <- B.readFile "mnist/train-images.idx3-ubyte"
  l <- B.readFile "mnist/train-labels.idx1-ubyte"
  return $ (i, l)

readTestDataSets :: IO (B.ByteString, B.ByteString)
readTestDataSets  = do
  downloadDatasetsIfMissing
  i <- B.readFile "mnist/test-images.idx3-ubyte"
  l <- B.readFile "mnist/test-labels.idx1-ubyte"
  return $ (i, l)

-- | downloadDatasetsIfMissing
--
-- Download mnist datast if missing
downloadDatasetsIfMissing :: IO ()
downloadDatasetsIfMissing = do
  ti1 <- doesFileExist "mnist/train-images.idx3-ubyte"
  tl1 <- doesFileExist "mnist/train-labels.idx1-ubyte"
  ti2 <- doesFileExist "mnist/test-images.idx3-ubyte"
  tl2 <- doesFileExist "mnist/test-labels.idx1-ubyte"
  if not ti1 || not tl1 || not ti2 || not tl2 then
    downloadAll
  else return ()
  where
    downloadAll = mapM_ f urls
    f (url, path) = do
      d <- download url
      BL.writeFile path d
    download = fmap (decompress. (^. responseBody)) . get
    urls = [ ( "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
             , "mnist/train-images.idx3-ubyte")
           , ( "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
             , "mnist/train-labels.idx1-ubyte")
           , ( "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
             , "mnist/test-images.idx3-ubyte")
           , ( "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
             , "mnist/test-labels.idx1-ubyte")]
