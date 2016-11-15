module Main where

import Prelude hiding (map, zipWith)
import Control.Monad (sequence)
import Data.Array.Repa
import Data.Array.Repa.Algorithms.Matrix
import Data.Foldable (foldlM)
import System.Random (randomRs, mkStdGen)

import Mnist (readTrainDataSets, DataSet, normalize)

type Matrix = Array U DIM2 Double

type Vector = Array U DIM1 Double

type Weight = Matrix

type Bias = Matrix

type ActivationFunction = Double -> Double

type Layer = (Weight, Bias, ActivationFunction)

type NN = [Layer]

type LossFunction = Vector -> Vector -> Double

main :: IO ()
main = do
  d <- dataset
  as <- mapM (performNNStep n) d
  print $ length $ filter id as
  where
    dataset = do
      datasets <- fmap (take 1000) readTrainDataSets
      return $ fmap (\(d, l) -> ( fromListUnboxed (ix2 1 784) (normalize d)
                                , l)) datasets

    performNNStep :: NN -> (Matrix, Int) -> IO Bool
    performNNStep n (d, l) = do
      z <- fmap (toList.softmax) $ performNN d n
      let a = snd . maximum . zip z $ [1..]
      return $ a == l
    n = [ (x1, b1, sigmonoid)
         , (x2, b2, sigmonoid)
         , (x3, b3, id) ]
    s1 = 28 * 28 * 50
    x1 = matrix (ix2 784 50) s1 ((-10), 10.0) 10
    b1 = matrix (ix2 1 50) 50 (0.0, 1.0) 10
    x2 = matrix (ix2 50 100) (50*100) ((-10), 100) 30
    b2 = matrix (ix2 1 100) 100 ((-10), 100) 20
    x3 = matrix (ix2 100 10) (100 * 10) ((-10), 100) 5
    b3 = matrix (ix2 1 10) 10 ((-10), 10) 100

meanSquaredError :: LossFunction
meanSquaredError x y = 0.5 * (sumAllS $ zipWith (\a b -> (a-b)^2) x y)

crossEntropyError :: LossFunction
crossEntropyError x y = (*(-1)) . sumAllS . zipWith (*) y $
                        map (log . (+delta)) x
  where
    delta = 0.1 ^ 7

pickle :: Show a => a -> FilePath -> IO ()
pickle a f = writeFile f . show $ a

unpickle :: Read a => FilePath -> IO a
unpickle = fmap read . readFile

matrix :: DIM2 -> Int -> (Double, Double) -> Int -> Matrix
matrix shape len range gen = fromListUnboxed shape $
                             take len . randomRs range $ mkStdGen gen

performNN :: Monad m => Matrix -> NN -> m Matrix
performNN input n = foldlM f input n
  where
    f :: Monad m => Matrix -> Layer -> m Matrix
    f i (w, b, a) = computeP . map a .
                    zipWith (+) b $ mmultS i w

nn :: NN
nn = [ ( fromListUnboxed (ix2 2 3) [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]
       , fromListUnboxed (ix2 1 3) [0.1, 0.2, 0.3]
       , sigmonoid )
     , ( fromListUnboxed (ix2 3 2) [0.1, 0.4, 0.2, 0.5, 0.3, 0.6]
       , fromListUnboxed (ix2 1 2) [0.1, 0.2]
       , sigmonoid )
     , ( fromListUnboxed (ix2 2 2) [0.1, 0.3, 0.2, 0.4]
       , fromListUnboxed (ix2 1 2) [0.1, 0.2]
       , id)
     ]

sigmonoid :: Double -> Double
sigmonoid x = 1 / (1 + exp (-x))

softmax :: Matrix -> Matrix
softmax x = let c = foldAllS max 0 x
                expA :: Matrix
                expA = computeS $ map (\a -> exp $ a - c) x
                sumExpA = sumAllS expA
            in computeS $ map (/sumExpA) expA
