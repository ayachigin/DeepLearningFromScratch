module Main where

import System.IO (stdout, hFlush)
import Control.Monad (foldM_)
import Data.Array.Repa hiding (map, zipWith, (++))
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (row, col, mmultS)

import Data.IORef (newIORef, readIORef, writeIORef)

import Mnist (randomNormalizedDataset, NormalizedDataSet)
import NeuralNetwork
import Optimizer (updateAdamGradMaybe, AdamGrad)
import Util
import Types

batsize :: Int
batsize = 500

learningRate :: Double
learningRate = 0.01

iterNum :: Int
iterNum = 60 * 10

main :: IO ()
main = do
  d <- dataset
  n <- network [28*28, 200, 100, 10] (-1, 2)
  p . ("Accuracy:"++) . format $ accuracy d batsize n
  p . ("Loss    :"++) . (show)  $ loss d batsize n
  let nb = nnb n
  m <- newIORef (Nothing :: Maybe AdamGrad)
  p . ("AccuracyB:"++) . format $ accuracyB d batsize nb
  p . ("LossB    :"++) . (show) . fst $ lossB d batsize nb
  p . length . fst . (!!1) $ gradient d batsize nb
  p . length . snd . (!!1) $ gradient d batsize nb
  foldM_ (\ni _ -> performNN m ni) nb [1..iterNum]
  where
    performNN adamgradRef n = do
      d <- dataset
      adamgrad <- readIORef adamgradRef
      let g = gradient d batsize n
          (n', m') = updateAdamGradMaybe learningRate adamgrad (nbn n) g
      --n' <- gradientDescentB learningRate n g
      writeIORef adamgradRef (Just m')
      p . ("Accuracy:"++) . format $ accuracyB d batsize n
      return $ nnb n'
    f (w, b, _) = (w, b)
    p s = print s >> hFlush stdout
    format = (++"%"). show . floor . (*100)
    dataset :: IO NormalizedDataSet
    dataset = randomNormalizedDataset batsize
