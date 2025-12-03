using ManyExpertDecisionTrees
using Test
using DataFrames
using FuzzyLogic

import DecisionTree: build_tree

@testset "ManyExpertDecisionTrees.jl" begin
    
    @testset "subdivide" begin
        df = DataFrame(a = 1:10, b = 11:20)
        
        parts = ManyExpertDecisionTrees.subdivide(3, df)
        @test length(parts) == 3
        @test size(parts[1], 1) == 3
        @test size(parts[2], 1) == 3
        @test size(parts[3], 1) == 4  
        
        parts = ManyExpertDecisionTrees.subdivide(1, df)
        @test length(parts) == 1
        @test size(parts[1], 1) == 10
    end
    
    @testset "get_params" begin
        df = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dfview = @view df[1 : end - 1, :]
        lp, rp = ManyExpertDecisionTrees.get_params(3.0, 1, dfview, GaussianMF)
        
        @test lp[1] ≈ 2.0  
        @test rp[1] ≈ 4.5  
        @test lp[2] > 0   
        @test rp[2] > 0
        
        @test_throws ErrorException ManyExpertDecisionTrees.get_params(
            1.0, 1, dfview, TriangularMF
        )
    end
    
   
    @testset "Leaf" begin
        leaf = ManyExpertDecisionTrees.MEDTLeaf(1);
        @test leaf.label == 1
        @test length(leaf) == 1
        @test depth(leaf) == 0
    end
    
    @testset "Node" begin
        leaf1 = ManyExpertDecisionTrees.MEDTLeaf(0)
        leaf2 = ManyExpertDecisionTrees.MEDTLeaf(1)
        
        mf1 = GaussianMF(0.0, 1.0)
        mf2 = GaussianMF(1.0, 1.0)
        
        node = ManyExpertDecisionTrees.MEDTNode(
            0.5, 1,
            (mf1,), (mf2,),
            leaf1, leaf2
        )
        
        @test node.featval == 0.5
        @test node.featid == 1
        @test length(node) == 2
        @test depth(node) == 1
    end
    
    @testset "manify" begin
        X = DataFrame(
            x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            x2 = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        )
        y = [0, 0, 1, 1, 1, 1]
        
        dt = build_tree(y, Matrix(X))
        
        experts = (GaussianMF, GaussianMF)
        medt = manify(dt, X, experts)
        
        @test medt isa ManyExpertDecisionTree{2}
        @test medt.featnames == ["x1", "x2"]
        @test length(medt.mftypes) == 2
        @test length(medt) >= 1
        @test depth(medt) >= 0
        
        leaf = ManyExpertDecisionTrees.MEDTLeaf(1)
        
        struct FakeMF{N}  
        end
        
        @test_throws ErrorException ManyExpertDecisionTrees.ManyExpertDecisionTree{1}(
            leaf, ["x"], (FakeMF,)
        )

    end     
end
