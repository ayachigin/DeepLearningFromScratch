module NeuralNetwork where

import Prelude hiding (map, zipWith)
import Data.Array.Repa hiding ((++))
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (mmultS)
import System.Random (mkStdGen, randomRs)

import Control.Lens

type Matrix r = Array r DIM2 Double

type Vector = Array U DIM1 Double

type Weight = Matrix U

type Bias = Matrix U

type ActivationFunction = Matrix D -> Matrix U

type Layer = (Weight, Bias, ActivationFunction)

type Gradient = (Weight, Bias)

type Gradients = [Gradient]

weight :: Layer -> Weight
weight = (^._1)

bias :: Layer -> Bias
bias = (^._2)

type NN = [Layer]

type LossFunction = Matrix U -> Matrix U -> Double

batsize :: Int
batsize = 100

numericalGradient :: Matrix U -> NN -> LossFunction -> Gradients
numericalGradient input net loss = undefined

numericalGradient' :: ([Double] -> Double) -> [Double] -> [Double]
numericalGradient' f (l:ls) = g f [] l ls
  where
    g f left val [] = [diff f left val []]
    g f left val rrs@(r:rs) = diff f left val rrs:
                              g f (left++[val]) r rs
    diff f l x r = (f (l ++ [x+h] ++ r) - f (l ++ [x-h] ++ r)) / 2*h
    h = 0.1^3

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

calcLoss :: LossFunction -> Matrix U -> Int -> Double
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

matrix :: DIM2 -> Int -> (Double, Double) -> Int -> Matrix U
matrix shape len range gen = fromListUnboxed shape $
                             take len . randomRs range $ mkStdGen gen

predict :: Matrix U -> NN -> Matrix U
predict input n = foldl f input n
  where
    f :: Matrix U -> Layer -> Matrix U
    f i (w, b, a) = a .
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
       , softmax)
     ]

sigmonoid :: ActivationFunction
sigmonoid = computeS.map (\x -> 1 / (1 + exp (-x)))

softmax :: ActivationFunction
softmax x = let c = foldAllS max 0 x
                expA :: Matrix U
                expA = computeS $ map (\a -> exp $ a - c) x
                sumExpA = sumAllS expA
            in computeS $ map (/sumExpA) expA
