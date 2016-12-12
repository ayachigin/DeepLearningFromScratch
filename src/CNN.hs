{-# LANGUAGE TypeOperators, TypeSynonymInstances, FlexibleInstances #-}
module CNN where

import Prelude hiding (map, (++), zipWith, traverse)
import Data.Array.Repa

import Types

data Forward

data Backward

type MiniBatchInput = Array D DIM4 Double

data Convolution = Convolution (Matrix D) (Matrix D) Int Int

data CNNLayer a = ConvolutionLayer Convolution
                | Pooling

data Im2ColOption = Im2ColOption
  { inputSize    :: Int
  , filterHeight :: Int
  , filterWidth  :: Int
  , strideSize   :: Int
  , padSize      :: Int
  } deriving (Show, Read, Eq)

im2col :: (Source r1 e, Source r2 e, Shape sh) =>
          Im2ColOption ->
          Array r1 sh e ->
          Array r2 DIM2 e
im2col = undefined

convolutionForward :: MiniBatchInput ->
                      Convolution ->
                      CNNLayer Backward
convolutionForward = undefined

-- | focus
--
-- It provides access to the array that reduced one dimension of
-- muitidimensional array by index.
focus :: (Source r e, Shape sh) =>
         (Array r (sh :. Int) e) ->
         Int ->
         Array D sh e
focus arr idx = fromFunction sh' f
  where (sh' :. _) = extent arr
        f sh = index arr (sh :. idx)

pad :: (Source r1 e, Num e) =>
       Int ->
       Array r1 DIM2 e -> Array D DIM2 e
pad p arr = fromFunction (ix2 (x+2*p) (y+2*p)) f
  where (Z:.x:.y) = extent arr
        f (Z:.x':.y')= if g then 0
                       else index arr (ix2 (x'-p) (y'-p))
          where g = x' < p || y' < p || x' >= x + p || y' >= y + p

test :: Array U DIM2 Float
test = computeS $ pad 2 x
  where x = fromListUnboxed (ix2 2 3) [(1::Float)..6]
