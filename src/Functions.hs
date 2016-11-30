module Functions
  ( meanSquaredError
  , crossEntropyError
  , sigmonoid
  , softmax)
  where

import Prelude hiding (map, zipWith, traverse)
import Data.Array.Repa hiding ((++))

type Matrix r = Array r DIM2 Double

meanSquaredError :: Matrix U -> Matrix U -> Int -> Double
meanSquaredError x y n = (0.5 * (sumAllS $ zipWith (\a b -> (a-b)^2) x y))/
                         fromIntegral n

-- $setup
-- >>> import Util ((=~))
-- >>> let y = fromListUnboxed (ix2 2 2) [0.6, 0.9, 0.2, 0.3]
-- >>> let t = fromListUnboxed (ix2 2 2) [0, 1, 1, 0]
-- >>> let x = fromListUnboxed (ix2 2 3) [(1::Double)..6]

{- | crossEntropyError
>>> crossEntropyError y t 2 =~ 0.857399 $ 3
True
-}
crossEntropyError :: Matrix U -> Matrix U -> Int -> Double
crossEntropyError y t n = ((*(-1)) . sumAllS . (*^t) $
                           map (log . (+delta)) y) / fromIntegral n
  where
    delta = 1e-4

sigmonoid :: Matrix D -> Matrix U
sigmonoid = computeS . map (\x -> 1 / (1 + exp (-x)))

{- | softmax
>>> 2 =~ sumAllS (softmax (delay x)) $ 2
True
-}
softmax :: Matrix D -> Matrix U
softmax x = computeS $ expY /^  sumExpY
  where
    maxX2 = traverse x id (f (foldS max 0 x))
    y = x -^ maxX2 -- For overflow countermeasure
    expY = map exp y
    sumExpY = traverse expY id (f (foldS (+) 0 expY))
    f arr _ (Z:.i:._) = index arr (ix1 i)
