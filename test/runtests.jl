using ManyExpertDecisionTrees
using Test
using DataFrames
using Statistics
using SoleLogics
import FuzzyLogic as FL

import DecisionTree: build_tree

@testset "ManyExpertDecisionTrees.jl" begin
    
    @testset "subdivide" begin
        X = [1:10 11:20]
        
        parts = ManyExpertDecisionTrees.subdivide(3, X)
        @test length(parts) == 3
        @test size(parts[1], 1) == 3
        @test size(parts[2], 1) == 3
        @test size(parts[3], 1) == 4  
        
        parts = ManyExpertDecisionTrees.subdivide(1, X)
        @test length(parts) == 1
        @test size(parts[1], 1) == 10
    end
    
    @testset "get_params" begin
        X = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6, 1)
        
        μ, σ = ManyExpertDecisionTrees.get_params(1, X, FL.GaussianMF)
        
        @test μ ≈ 3.5  
        @test σ > 0    
        @test σ ≈ std([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Test error for unsupported membership function
        @test_throws ErrorException ManyExpertDecisionTrees.get_params(
            1, X, FL.TriangularMF
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
        X = [1.0 1.0; 2.0 1.0; 3.0 2.0; 4.0 2.0; 5.0 3.0; 6.0 3.0]
        y = [0, 0, 1, 1, 1, 1]
        
        dt = build_tree(y, X)
        
        medt = manify(dt, X, FL.GaussianMF, FL.GaussianMF)
        
        @test medt isa ManyExpertDecisionTree
        @test medt.nfeats == 2  
        @test length(medt.mftypes) == 2
        @test length(medt) >= 1
        @test depth(medt) >= 0
        
        leaf = ManyExpertDecisionTrees.MEDTLeaf(1)
        
        struct FakeMF{N}  
        end
        
        @test_throws ErrorException ManyExpertDecisionTrees.ManyExpertDecisionTree(leaf, 2, FakeMF, FakeMF)

    end
    
    @testset "apply" begin
        using SoleLogics.ManyValuedLogics
        
        X = [1.0 1.0; 2.0 1.0; 3.0 2.0; 4.0 2.0; 5.0 3.0; 6.0 3.0]
        y = [0, 0, 1, 1, 1, 1]
        
        dt = build_tree(y, X)
        medt = manify(dt, X, FL.GaussianMF, FL.GaussianMF)
        
        MXA = ManyExpertAlgebra(GodelLogic, GodelLogic)
        
        instance1 = [1.5, 1.0]
        result1 = apply(medt, MXA, instance1)
        
        @test result1 isa Vector
        @test !isempty(result1)
        
        instance2 = [5.0, 3.0]
        result2 = apply(medt, MXA, instance2)
        
        @test result2 isa Vector
        @test !isempty(result2)
        
        wrong_dim_instance = [1.0]  
        @test_throws ErrorException apply(medt, MXA, wrong_dim_instance)
        
        MXA_wrong = ManyExpertAlgebra(GodelLogic)
        @test_throws ErrorException apply(medt, MXA_wrong, instance1)
        
        instance_int = [2, 2]
        result_int = apply(medt, MXA, instance_int)
        @test result_int isa Vector
        
        medt3 = manify(dt, X, FL.GaussianMF, FL.GaussianMF, FL.GaussianMF)
        MXA3 = ManyExpertAlgebra(GodelLogic, ProductLogic, LukasiewiczLogic)
        
        result3 = apply(medt3, MXA3, instance1)
        @test result3 isa Vector
        @test !isempty(result3)

        medt4 = fuzzify(dt, X, FL.GaussianMF)
        MXA4 = ManyExpertAlgebra(GodelLogic)
        result4 = apply(medt4, MXA4, instance1)
        @test result4 isa Vector
        @test !isempty(result4)
    end
    
    # TODO: Re-enable when addexperts! is reimplemented
    # @testset "addexperts!" begin
    #     X = DataFrame(
    #         x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    #         x2 = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    #     )
    #     y = [0, 0, 1, 1, 1, 1]
    #     
    #     dt = build_tree(y, Matrix(X))
    #     X_matrix = Matrix(X)
    #     medt = manify(dt, X_matrix, FL.GaussianMF, FL.GaussianMF)
    #     
    #     initial_expert_count = length(medt.mftypes)
    #     initial_length = length(medt)
    #     
    #     addexperts!(medt, X_matrix, FL.GaussianMF)
    #     
    #     @test length(medt.mftypes) == initial_expert_count + 1
    #     @test length(medt) == initial_length  # Tree structure unchanged
    #     @test medt.mftypes[end] == FL.GaussianMF{Float64}
    #     
    #     root = medt.root
    #     if root isa ManyExpertDecisionTrees.MEDTNode
    #         @test length(root.mfleft) == initial_expert_count + 1
    #         @test length(root.mfright) == initial_expert_count + 1
    #     end
    #     
    #     addexperts!(medt, X_matrix, FL.GaussianMF, FL.GaussianMF)
    #     
    #     @test length(medt.mftypes) == initial_expert_count + 3
    #     @test medt.mftypes[end-1] == FL.GaussianMF{Float64}
    #     @test medt.mftypes[end] == FL.GaussianMF{Float64}
    #     
    #     if medt.root isa ManyExpertDecisionTrees.MEDTNode
    #         @test length(medt.root.mfleft) == initial_expert_count + 3
    #         @test length(medt.root.mfright) == initial_expert_count + 3
    #     end
    # end     
end
