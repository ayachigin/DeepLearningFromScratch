module Main where

import Prelude hiding (map, zipWith)
import Data.Array.Repa hiding ((++))
import Data.Array.Repa.Algorithms.Matrix (mmultS)
import Data.Array.Repa.Repr.Vector (V, computeVectorP, fromListVector)
import System.Random (randomRs, mkStdGen)

import Mnist (choiceIO, toArray, normalize, Label)

type Matrix = Array U DIM2 Double

type Vector = Array U DIM1 Double

type Weight = Matrix

type Bias = Matrix

type ActivationFunction = Double -> Double

type Layer = (Weight, Bias, ActivationFunction)

type NN = [Layer]

type LossFunction = Matrix -> Matrix -> Double

batsize :: Int
batsize = 100

main :: IO ()
main = do
  d <- dataset
  as <- computeVectorP $ map (performNNStep network) d
  print $ (/(fromIntegral batsize)) . sum . toList $ as
  where
    dataset :: IO (Array V DIM1 (Array U DIM2 Double, Label))
    dataset = do
      datasets <- fmap (normalize) $ choiceIO batsize
      return . toArray batsize $ datasets
    performNNStep :: NN -> (Matrix, Int) -> Double
    performNNStep n (d, l) =
      let z = softmax $ performNN d n
      in calcLoss crossEntropyError z l
    network = [ (x1, b1, sigmonoid)
              , (x2, b2, sigmonoid)
              , (x3, b3, id) ]
    s1 = 28 * 28 * 50
    x1 = matrix (ix2 784 50) s1 ((-10), 10.0) 10
    b1 = matrix (ix2 1 50) 50 (0.0, 1.0) 10
    x2 = matrix (ix2 50 100) (50*100) ((-10), 100) 30
    b2 = matrix (ix2 1 100) 100 ((-10), 100) 20
    x3 = matrix (ix2 100 10) (100 * 10) ((-10), 100) 5
    b3 = matrix (ix2 1 10) 10 ((-10), 10) 100

{- | loass function meanSquaredError
>>> let y = fromListUnboxed (ix2 1 10) [0.1, 0.05, 0.1, 0, 0.05, 0.1, 0, 0.6, 0, 0]
>>> let t = fromListUnboxed (ix2 1 10) [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>> meanSquaredError y t
0.5974999999999999
>>> let y = fromListUnboxed (ix2 1 10) [0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0]
>>> meanSquaredError y t
9.750000000000003e-2
-}
meanSquaredError :: LossFunction
meanSquaredError x y = 0.5 * (sumAllS $ zipWith (\a b -> (a-b)^2) x y)

crossEntropyError :: LossFunction
crossEntropyError x y = (*(-1)) . sumAllS . zipWith (*) y $
                        map (log . (+delta)) x
  where
    delta = 0.1 ^ 7

calcLoss :: LossFunction -> Matrix -> Int -> Double
calcLoss f m i = f m $ arr i
  where
    arr = fromListUnboxed (ix2 1 10) . shift bits
    shift [] _ = error "Shift empty list"
    shift ls 0 = ls
    shift (x:xs) n = shift (xs ++ [x]) (n-1)
    bits = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



pickle :: Show a => a -> FilePath -> IO ()
pickle a f = writeFile f . show $ a

unpickle :: Read a => FilePath -> IO a
unpickle = fmap read . readFile

matrix :: DIM2 -> Int -> (Double, Double) -> Int -> Matrix
matrix shape len range gen = fromListUnboxed shape $
                             take len . randomRs range $ mkStdGen gen

performNN :: Matrix -> NN -> Matrix
performNN input n = foldl f input n
  where
    f :: Matrix -> Layer -> Matrix
    f i (w, b, a) = computeS . map a .
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
