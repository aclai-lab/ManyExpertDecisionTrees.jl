using ManyExpertDecisionTrees
using Test
using DataFrames
import FuzzyLogic as FL

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
        lp, rp = ManyExpertDecisionTrees.get_params(3.0, 1, dfview, FL.GaussianMF)
        
        @test lp[1] ≈ 2.0  
        @test rp[1] ≈ 4.5  
        @test lp[2] > 0   
        @test rp[2] > 0
        
        @test_throws ErrorException ManyExpertDecisionTrees.get_params(
            1.0, 1, dfview, FL.TriangularMF
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
        
        node = ManyExpertDecisionTrees.MEDTNode(
            0.5, 
            1,
            FL.AbstractMembershipFunction[FL.GaussianMF(0.0,1.0)], 
            FL.AbstractMembershipFunction[FL.GaussianMF(0.0, 1.0)],
            leaf1, 
            leaf2
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
        
       
        medt = manify(dt, X, FL.GaussianMF, FL.GaussianMF)
        
        @test medt isa ManyExpertDecisionTree
        @test medt.featnames == ["x1", "x2"]
        @test length(medt.mftypes) == 2
        @test length(medt) >= 1
        @test depth(medt) >= 0
        
        leaf = ManyExpertDecisionTrees.MEDTLeaf(1)
        
        struct FakeMF{N}  
        end
        
        @test_throws ErrorException ManyExpertDecisionTrees.ManyExpertDecisionTree(leaf, ["x"], FakeMF, FakeMF)

    end
    
    @testset "addexperts!" begin
        X = DataFrame(
            x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            x2 = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        )
        y = [0, 0, 1, 1, 1, 1]
        
        dt = build_tree(y, Matrix(X))
        medt = manify(dt, X, FL.GaussianMF, FL.GaussianMF)
        
        initial_expert_count = length(medt.mftypes)
        initial_length = length(medt)
        
        addexperts!(medt, X, FL.GaussianMF)
        
        @test length(medt.mftypes) == initial_expert_count + 1
        @test length(medt) == initial_length  # Tree structure unchanged
        @test medt.mftypes[end] == FL.GaussianMF{Float64}
        
        root = medt.root
        if root isa ManyExpertDecisionTrees.MEDTNode
            @test length(root.mfleft) == initial_expert_count + 1
            @test length(root.mfright) == initial_expert_count + 1
        end
        
        addexperts!(medt, X, FL.GaussianMF, FL.GaussianMF)
        
        @test length(medt.mftypes) == initial_expert_count + 3
        @test medt.mftypes[end-1] == FL.GaussianMF{Float64}
        @test medt.mftypes[end] == FL.GaussianMF{Float64}
        
        if medt.root isa ManyExpertDecisionTrees.MEDTNode
            @test length(medt.root.mfleft) == initial_expert_count + 3
            @test length(medt.root.mfright) == initial_expert_count + 3
        end
    end     
end
