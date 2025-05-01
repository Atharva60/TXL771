import streamlit as st
import numpy as np
import pandas as pd
import math
from collections import defaultdict

# ==============================
# Core Micromechanics Functions
# ==============================

def micro(E1f, E2f, v12f, Em, vm, theta, Vf):
    E1 = (float(E1f) * float(Vf)) + (float(Em) * (1 - float(Vf)))
    E2 = 1.0 / ((float(Vf)/float(E2f)) + ((1.0 - float(Vf))/float(Em)))
    G12f = float(E1f) / (2.0 * (1.0 + float(v12f)))
    Gm = float(Em) / (2.0 * (1.0 + float(vm)))
    G12 = 1.0 / ((float(Vf)/G12f) + ((1.0 - float(Vf))/Gm))
    v12 = float(v12f) * float(Vf) + float(vm) * (1.0 - float(Vf))
    return E1, E2, G12, v12

def Q_bar(E1, E2, G12, v12, theta):
    Q_inv = np.array([
        [1.0/float(E1), -float(v12)/float(E1), 0.0],
        [-float(v12)/float(E1), 1.0/float(E2), 0.0],
        [0.0, 0.0, 1.0/float(G12)]
    ], dtype=np.float64)
    
    Q = np.linalg.inv(Q_inv)
    theta_rad = math.radians(float(theta))
    s = math.sin(theta_rad)
    c = math.cos(theta_rad)
    
    T = np.array([
        [c**2, s**2, 2*s*c],
        [s**2, c**2, -2*s*c],
        [-s*c, s*c, c**2 - s**2]
    ], dtype=np.float64)
    
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
    T_inv = np.linalg.inv(T)
    R_inv = np.linalg.inv(R)
    
    Q_bar = T_inv @ Q @ R @ T @ R_inv
    return Q_bar.astype(np.float64)

def find_ABD(n, layers, material_db):
    given_data = np.zeros((n, 6), dtype=np.float64)
    Q_bar_data = np.zeros((n, 3, 3), dtype=np.float64)
    
    for i in range(n):
        layer = layers[i]
        mat = material_db[layer['material']]
        
        if mat['type'] == "Fiber/Matrix":
            props = mat['props']
            Vf = float(layer.get('Vf', 0.6))
            
            E1, E2, G12, v12 = micro(
                props['E1f'], props['E2f'], props['v12f'],
                props['Em'], props['vm'],
                float(layer['angle']), Vf
            )
        else:
            props = mat['props']
            E1 = props['E1']
            E2 = props['E2']
            G12 = props['G12']
            v12 = props['v12']
        
        given_data[i, 0] = E1
        given_data[i, 1] = E2
        given_data[i, 2] = G12
        given_data[i, 3] = v12
        given_data[i, 4] = float(layer['angle'])
        given_data[i, 5] = float(layer['thickness'])
        
        Q_bar_data[i] = Q_bar(E1, E2, G12, v12, layer['angle'])
    
    lower = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    
    upper[0] = lower[0] + given_data[0, 5]
    for i in range(1, n):
        lower[i] = upper[i-1]
        upper[i] = lower[i] + given_data[i, 5]
    
    midpoint = upper[-1] / 2.0
    lower -= midpoint
    upper -= midpoint
    
    ABD = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(n):
                ABD[i, j] += Q_bar_data[k, i, j] * (upper[k] - lower[k])
                ABD[i+3, j] += 0.5 * Q_bar_data[k, i, j] * (upper[k]**2 - lower[k]**2)
                ABD[i, j+3] += 0.5 * Q_bar_data[k, i, j] * (upper[k]**2 - lower[k]**2)
                ABD[i+3, j+3] += (1.0/3.0) * Q_bar_data[k, i, j] * (upper[k]**3 - lower[k]**3)
    
    ABD_inv = np.linalg.inv(ABD)
    return ABD, ABD_inv, Q_bar_data, lower, upper, given_data

def stress_strain(n, layers, material_db, loading):
    ABD, ABD_inv, Q_bar_data, lower, upper, given_data = find_ABD(n, layers, material_db)
    
    force_vector = np.array([
        [loading['NX']],
        [loading['NY']],
        [loading['NXY']],
        [loading['MX']],
        [loading['MY']],
        [loading['MXY']]
    ], dtype=np.float64)
    
    strain_curvature = ABD_inv @ force_vector
    
    # Strain calculations
    Layer_strain_lower = np.zeros((n, 3), dtype=np.float64)
    Layer_strain_upper = np.zeros((n, 3), dtype=np.float64)
    Layer_strain_middle = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
        z_lower = lower[i]
        z_upper = upper[i]
        z_mid = (z_lower + z_upper) / 2.0
        
        Layer_strain_lower[i] = (strain_curvature[:3] + z_lower * strain_curvature[3:]).flatten()
        Layer_strain_upper[i] = (strain_curvature[:3] + z_upper * strain_curvature[3:]).flatten()
        Layer_strain_middle[i] = (strain_curvature[:3] + z_mid * strain_curvature[3:]).flatten()
    
    # Stress calculations
    Layer_stress_lower = np.array([Q_bar_data[i] @ Layer_strain_lower[i] for i in range(n)])
    Layer_stress_upper = np.array([Q_bar_data[i] @ Layer_strain_upper[i] for i in range(n)])
    Layer_stress_middle = np.array([Q_bar_data[i] @ Layer_strain_middle[i] for i in range(n)])
    
    # Transformation to fiber coordinates
    fiber_stress_lower = np.zeros((n, 3), dtype=np.float64)
    fiber_stress_upper = np.zeros((n, 3), dtype=np.float64)
    fiber_stress_middle = np.zeros((n, 3), dtype=np.float64)
    
    fiber_strain_lower = np.zeros((n, 3), dtype=np.float64)
    fiber_strain_upper = np.zeros((n, 3), dtype=np.float64)
    fiber_strain_middle = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
        theta = math.radians(given_data[i, 4])
        c = math.cos(theta)
        s = math.sin(theta)
        
        T = np.array([
            [c**2, s**2, 2*s*c],
            [s**2, c**2, -2*s*c],
            [-s*c, s*c, c**2 - s**2]
        ], dtype=np.float64)
        
        fiber_stress_lower[i] = T @ Layer_stress_lower[i]
        fiber_stress_upper[i] = T @ Layer_stress_upper[i]
        fiber_stress_middle[i] = T @ Layer_stress_middle[i]
        
        T_strain = np.array([
            [c**2, s**2, s*c],
            [s**2, c**2, -s*c],
            [-2*s*c, 2*s*c, c**2 - s**2]
        ], dtype=np.float64)
        
        fiber_strain_lower[i] = T_strain @ Layer_strain_lower[i]
        fiber_strain_upper[i] = T_strain @ Layer_strain_upper[i]
        fiber_strain_middle[i] = T_strain @ Layer_strain_middle[i]
    
    # Calculate laminate moduli
    midplane_strain = strain_curvature[:3].flatten()
    total_thickness = np.sum(given_data[:, 5])
    
    Ex = 0
    if abs(midplane_strain[0]) > 1e-10:  # Avoid division by zero
        Ex = loading['NX'] / (total_thickness * midplane_strain[0])
    
    Ey = 0
    if abs(midplane_strain[1]) > 1e-10:
        Ey = loading['NY'] / (total_thickness * midplane_strain[1])
    
    Gxy = 0
    if abs(midplane_strain[2]) > 1e-10:
        Gxy = loading['NXY'] / (total_thickness * midplane_strain[2])
    
    # Collect max stress locations
    max_stress_x_idx = np.argmax(np.abs(np.concatenate([Layer_stress_upper[:, 0], Layer_stress_lower[:, 0]])))
    max_stress_y_idx = np.argmax(np.abs(np.concatenate([Layer_stress_upper[:, 1], Layer_stress_lower[:, 1]])))
    max_stress_xy_idx = np.argmax(np.abs(np.concatenate([Layer_stress_upper[:, 2], Layer_stress_lower[:, 2]])))
    
    layer_idx_x = max_stress_x_idx % n
    layer_idx_y = max_stress_y_idx % n
    layer_idx_xy = max_stress_xy_idx % n
    
    is_upper_x = max_stress_x_idx < n
    is_upper_y = max_stress_y_idx < n
    is_upper_xy = max_stress_xy_idx < n
    
    max_stress_locations = {
        'x': {'layer': layer_idx_x + 1, 'surface': 'Upper' if is_upper_x else 'Lower'},
        'y': {'layer': layer_idx_y + 1, 'surface': 'Upper' if is_upper_y else 'Lower'},
        'xy': {'layer': layer_idx_xy + 1, 'surface': 'Upper' if is_upper_xy else 'Lower'}
    }
    
    return (
        Layer_stress_lower, Layer_stress_upper, Layer_strain_lower, Layer_strain_upper,
        Layer_stress_middle, Layer_strain_middle, fiber_stress_lower, fiber_stress_upper,
        fiber_stress_middle, given_data, ABD, ABD_inv, Q_bar_data, strain_curvature,
        fiber_strain_lower, fiber_strain_upper, fiber_strain_middle,
        Ex, Ey, Gxy, max_stress_locations
    )

def calculate_safety_factors(stresses, strengths):
    safety_factors = []
    failure_indices = []
    failure_modes = []
    
    for stress, strength in zip(stresses, strengths):
        Xt, Yt, Xc, Yc, S = strength
        sig1, sig2, tau = stress
        
        f1 = sig1/Xt if sig1 > 0 else -sig1/Xc
        f2 = sig2/Yt if sig2 > 0 else -sig2/Yc
        fs = abs(tau)/S
        
        max_f = max(f1, f2, fs)
        sf = 1.0 / max_f if max_f > 0 else float('inf')
        
        mode = 'Fiber Tension' if f1 == max_f and sig1 > 0 else (
               'Fiber Compression' if f1 == max_f and sig1 <= 0 else (
               'Matrix Tension' if f2 == max_f and sig2 > 0 else (
               'Matrix Compression' if f2 == max_f and sig2 <= 0 else 'Shear')))
        
        safety_factors.append(sf)
        failure_indices.append(max_f)
        failure_modes.append(mode)
    
    return safety_factors, failure_indices, failure_modes

# ==============================
# Streamlit Interface
# ==============================

def main():
    st.set_page_config(page_title="Composite Analyzer", layout="wide")
    st.title("Advanced Composite Laminate Analysis")
    
    # Initialize session state
    if 'materials' not in st.session_state:
        st.session_state.materials = defaultdict(dict)
    if 'layers' not in st.session_state:
        st.session_state.layers = []
    if 'loading' not in st.session_state:
        st.session_state.loading = {
            'NX': 0.0, 'NY': 0.0, 'NXY': 0.0,
            'MX': 0.0, 'MY': 0.0, 'MXY': 0.0
        }
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Material Database
    with st.sidebar.expander("🧪 Material Database", expanded=True):
        mat_name = st.text_input("Material Name")
        mat_type = st.radio("Type", ["Fiber/Matrix", "Composite"])
        
        col1, col2 = st.columns(2)
        props = {}
        with col1:
            if mat_type == "Fiber/Matrix":
                props['E1f'] = st.number_input("E1 Fiber (GPa)", value=230.0)
                props['E2f'] = st.number_input("E2 Fiber (GPa)", value=15.0)
                props['v12f'] = st.number_input("ν12 Fiber", value=0.3)
                props['Em'] = st.number_input("E Matrix (GPa)", value=3.5)
                props['vm'] = st.number_input("ν Matrix", value=0.35)
            else:
                props['E1'] = st.number_input("E1 Composite (GPa)", value=120.0)
                props['E2'] = st.number_input("E2 Composite (GPa)", value=8.0)
                props['G12'] = st.number_input("G12 (GPa)", value=4.5)
                props['v12'] = st.number_input("ν12", value=0.3)
        
        with col2:
            st.subheader("Strength Properties")
            strength = {
                'Xt': st.number_input("Xt (MPa)", value=2500.0),
                'Yt': st.number_input("Yt (MPa)", value=60.0),
                'Xc': st.number_input("Xc (MPa)", value=1800.0),
                'Yc': st.number_input("Yc (MPa)", value=120.0),
                'S': st.number_input("S (MPa)", value=80.0)
            }
        
        if st.button("💾 Save Material"):
            st.session_state.materials[mat_name] = {
                'type': mat_type,
                'props': {k: float(v) for k, v in props.items()},
                'strength': {k: float(v) for k, v in strength.items()}
            }
            st.success(f"Material {mat_name} saved!")

    # Loading Conditions
    with st.sidebar.expander("📈 Loading Conditions"):
        st.session_state.loading['NX'] = st.number_input("NX (N/mm)", value=0.0)
        st.session_state.loading['NY'] = st.number_input("NY (N/mm)", value=0.0)
        st.session_state.loading['NXY'] = st.number_input("NXY (N/mm)", value=0.0)
        st.session_state.loading['MX'] = st.number_input("MX (N·mm/mm)", value=0.0)
        st.session_state.loading['MY'] = st.number_input("MY (N·mm/mm)", value=0.0)
        st.session_state.loading['MXY'] = st.number_input("MXY (N·mm/mm)", value=0.0)

    # Layer Stacking
    with st.sidebar.expander("🧱 Layer Stacking"):
        if st.session_state.materials:
            mat_choice = st.selectbox("Select Material", list(st.session_state.materials.keys()))
            angle = st.slider("Ply Angle (°)", -90, 90, 0)
            thickness = st.number_input("Thickness (mm)", 0.01, 10.0, 0.2)
            
            if mat_choice in st.session_state.materials and st.session_state.materials[mat_choice]['type'] == "Fiber/Matrix":
                Vf = st.slider("Fiber Volume Fraction", 0.1, 0.9, 0.6, 0.01)
            else:
                Vf = 0.6
            
            if st.button("➕ Add Layer"):
                new_layer = {
                    'material': mat_choice,
                    'angle': angle,
                    'thickness': thickness
                }
                if st.session_state.materials[mat_choice]['type'] == "Fiber/Matrix":
                    new_layer['Vf'] = Vf
                st.session_state.layers.append(new_layer)
        else:
            st.warning("Create materials first!")

    # Main Interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Current Stack")
        if st.session_state.layers:
            for i, layer in enumerate(reversed(st.session_state.layers)):
                extra_info = f"Vf: {layer.get('Vf', 0.6)}" if 'Vf' in layer else ""
                st.markdown(f"""
                **Layer {len(st.session_state.layers)-i}**  
                Material: {layer['material']}  
                Angle: {layer['angle']}°  
                Thickness: {layer['thickness']}mm  
                {extra_info}
                """)
                if st.button(f"Remove Layer {len(st.session_state.layers)-i}", key=f"remove_{i}"):
                    del st.session_state.layers[len(st.session_state.layers)-i-1]
                    st.rerun()
        else:
            st.info("No layers in stack")

    with col2:
        if st.button("🚀 Run Analysis"):
            if not st.session_state.layers:
                st.error("Add layers first!")
            else:
                try:
                    # Run analysis
                    analysis_results = stress_strain(
                        n=len(st.session_state.layers),
                        layers=st.session_state.layers,
                        material_db=st.session_state.materials,
                        loading=st.session_state.loading
                    )
                    st.session_state.analysis_results = analysis_results
                    st.success("Analysis completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"🚨 Analysis failed: {str(e)}")
        
        if st.session_state.analysis_results:
            # Unpack results
            (Layer_stress_lower, Layer_stress_upper, Layer_strain_lower, Layer_strain_upper,
             Layer_stress_middle, Layer_strain_middle, fiber_stress_lower, fiber_stress_upper,
             fiber_stress_middle, given_data, ABD, ABD_inv, Q_bar_data, strain_curvature,
             fiber_strain_lower, fiber_strain_upper, fiber_strain_middle,
             Ex, Ey, Gxy, max_stress_locations) = st.session_state.analysis_results
            
            # Get material strengths
            strengths = [
                [
                    float(st.session_state.materials[layer['material']]['strength']['Xt']),
                    float(st.session_state.materials[layer['material']]['strength']['Yt']),
                    float(st.session_state.materials[layer['material']]['strength']['Xc']),
                    float(st.session_state.materials[layer['material']]['strength']['Yc']),
                    float(st.session_state.materials[layer['material']]['strength']['S'])
                ]
                for layer in st.session_state.layers
            ]
            
            # Calculate safety factors
            sf_lower, fi_lower, mode_lower = calculate_safety_factors(fiber_stress_lower, strengths)
            sf_upper, fi_upper, mode_upper = calculate_safety_factors(fiber_stress_upper, strengths)
            
            # Create tabs for different result sections
            analysis_tabs = st.tabs([
                "Stiffness Matrix", "Stresses & Strains", "Failure Analysis", 
                "Laminate Properties", "Layer Properties"
            ])
            
            # Tab 1: Stiffness Matrix
            with analysis_tabs[0]:
                st.header("Stiffness Matrices")
                
                st.subheader("ABD Matrix")
                ABD_df = pd.DataFrame(ABD, 
                    columns=['Nx', 'Ny', 'Nxy', 'Mx', 'My', 'Mxy'],
                    index=['Nx', 'Ny', 'Nxy', 'Mx', 'My', 'Mxy'])
                st.dataframe(ABD_df)
                
                st.subheader("Q-bar Matrices")
                q_bar_tabs = st.tabs([f"Layer {i+1}" for i in range(len(st.session_state.layers))])
                for i, tab in enumerate(q_bar_tabs):
                    with tab:
                        Q_bar_df = pd.DataFrame(
                            Q_bar_data[i],
                            columns=['1', '2', '6'],
                            index=['1', '2', '6']
                        )
                        st.dataframe(Q_bar_df)
            
            # Tab 2: Stresses & Strains
            with analysis_tabs[1]:
                st.header("Stresses & Strains")
                
                # Create a selection for data type
                vis_type = st.radio(
                    "Data Type",
                    ["Stress (Global)", "Stress (Fiber)", "Strain (Global)", "Strain (Fiber)"],
                    horizontal=True
                )
                
                component = st.radio(
                    "Component",
                    ["x/1-direction", "y/2-direction", "xy/12-direction"],
                    horizontal=True
                )
                
                # Mapping for indexing
                comp_idx = 0 if component == "x/1-direction" else (1 if component == "y/2-direction" else 2)
                
                # Show numerical values in tables
                with st.expander("Data Tables", expanded=True):
                    data_tabs = st.tabs(["Upper Surface", "Middle", "Lower Surface"])
                    
                    # Create dataframes for each location
                    if vis_type == "Stress (Global)":
                        df_upper = pd.DataFrame(Layer_stress_upper, columns=["σx", "σy", "τxy"])
                        df_middle = pd.DataFrame(Layer_stress_middle, columns=["σx", "σy", "τxy"])
                        df_lower = pd.DataFrame(Layer_stress_lower, columns=["σx", "σy", "τxy"])
                    elif vis_type == "Stress (Fiber)":
                        df_upper = pd.DataFrame(fiber_stress_upper, columns=["σ1", "σ2", "τ12"])
                        df_middle = pd.DataFrame(fiber_stress_middle, columns=["σ1", "σ2", "τ12"])
                        df_lower = pd.DataFrame(fiber_stress_lower, columns=["σ1", "σ2", "τ12"])
                    elif vis_type == "Strain (Global)":
                        df_upper = pd.DataFrame(Layer_strain_upper * 1e6, columns=["εx", "εy", "γxy"])
                        df_middle = pd.DataFrame(Layer_strain_middle * 1e6, columns=["εx", "εy", "γxy"])
                        df_lower = pd.DataFrame(Layer_strain_lower * 1e6, columns=["εx", "εy", "γxy"])
                    else:  # Strain (Fiber)
                        df_upper = pd.DataFrame(fiber_strain_upper * 1e6, columns=["ε1", "ε2", "γ12"])
                        df_middle = pd.DataFrame(fiber_strain_middle * 1e6, columns=["ε1", "ε2", "γ12"])
                        df_lower = pd.DataFrame(fiber_strain_lower * 1e6, columns=["ε1", "ε2", "γ12"])
                    
                    # Add layer index
                    layers = range(1, len(st.session_state.layers)+1)
                    df_upper.insert(0, "Layer", layers)
                    df_middle.insert(0, "Layer", layers)
                    df_lower.insert(0, "Layer", layers)
                    
                    # Display in tabs
                    with data_tabs[0]:
                        st.dataframe(df_upper)
                    with data_tabs[1]:
                        st.dataframe(df_middle)
                    with data_tabs[2]:
                        st.dataframe(df_lower)
                
                # Through-thickness numerical data
                st.subheader("Layer-by-Layer Data")
                
                # Calculate z-coordinates
                thickness_values = given_data[:, 5]
                cum_thickness = np.cumsum(thickness_values)
                total_thickness = cum_thickness[-1]
                midpoint = total_thickness / 2.0
                
                z_coords = []
                for i in range(len(st.session_state.layers)):
                    if i == 0:
                        lower_z = 0
                    else:
                        lower_z = cum_thickness[i-1]
                    upper_z = cum_thickness[i]
                    
                    # Adjust to be centered around midpoint
                    lower_z -= midpoint
                    upper_z -= midpoint
                    
                    z_coords.append((lower_z, upper_z))
                
                # Create a table with z-coordinate information
                z_data = []
                for i, (lower_z, upper_z) in enumerate(z_coords):
                    z_data.append([
                        f"Layer {i+1}",
                        f"{lower_z:.3f}",
                        f"{upper_z:.3f}",
                        f"{upper_z - lower_z:.3f}",
                        f"{st.session_state.layers[i]['angle']}°"
                    ])
                
                z_df = pd.DataFrame(z_data, columns=["Layer", "Lower z (mm)", "Upper z (mm)", "Thickness (mm)", "Orientation"])
                st.dataframe(z_df)
            
            # Tab 3: Failure Analysis
            with analysis_tabs[2]:
                st.header("Failure Analysis")
                
                # Failure status table
                st.subheader("Failure Status")
                
                failure_data = []
                for i in range(len(st.session_state.layers)):
                    upper_status = "Failed" if fi_upper[i] >= 1.0 else "Safe"
                    lower_status = "Failed" if fi_lower[i] >= 1.0 else "Safe"
                    
                    failure_data.append([
                        f"Layer {i+1}",
                        f"{sf_upper[i]:.2f}",
                        mode_upper[i],
                        upper_status,
                        f"{sf_lower[i]:.2f}",
                        mode_lower[i],
                        lower_status
                    ])
                
                st.table(pd.DataFrame(failure_data, 
                    columns=["Layer", "Upper SF", "Upper Failure Mode", "Upper Status", 
                             "Lower SF", "Lower Failure Mode", "Lower Status"]))
                
                # Summary of failure
                st.subheader("Failure Summary")
                
                min_sf = min(min(sf_upper), min(sf_lower))
                min_layer_idx = -1
                min_surface = ""
                min_mode = ""
                
                for i in range(len(st.session_state.layers)):
                    if sf_upper[i] == min_sf:
                        min_layer_idx = i
                        min_surface = "Upper"
                        min_mode = mode_upper[i]
                    elif sf_lower[i] == min_sf:
                        min_layer_idx = i
                        min_surface = "Lower"
                        min_mode = mode_lower[i]
                
                if min_sf < 1.0:
                    st.error(f"Laminate fails with minimum safety factor of {min_sf:.3f} in Layer {min_layer_idx+1} ({min_surface} surface) due to {min_mode}.")
                else:
                    st.success(f"Laminate is safe with minimum safety factor of {min_sf:.3f} in Layer {min_layer_idx+1} ({min_surface} surface).")
                
                # Failure indices table
                st.subheader("Failure Indices")
                
                fi_data = []
                for i in range(len(st.session_state.layers)):
                    fi_data.append([
                        f"Layer {i+1}",
                        f"{fi_upper[i]:.3f}",
                        f"{fi_lower[i]:.3f}",
                    ])
                
                st.table(pd.DataFrame(fi_data, 
                    columns=["Layer", "Upper Surface FI", "Lower Surface FI"]))
            
            # Tab 4: Laminate Properties
            with analysis_tabs[3]:
                st.header("Laminate Properties")
                
                # Display effective laminate moduli
                st.subheader("Effective Laminate Moduli")
                moduli_df = pd.DataFrame([
                    ["Ex (GPa)", f"{Ex/1000:.2f}"],
                    ["Ey (GPa)", f"{Ey/1000:.2f}"],
                    ["Gxy (GPa)", f"{Gxy/1000:.2f}"]
                ], columns=["Property", "Value"])
                st.table(moduli_df)
                
                # Display midplane strains and curvatures
                st.subheader("Midplane Deformation")
                midplane_df = pd.DataFrame([
                    ["εx (με)", f"{strain_curvature[0][0]*1e6:.2f}"],
                    ["εy (με)", f"{strain_curvature[1][0]*1e6:.2f}"],
                    ["γxy (με)", f"{strain_curvature[2][0]*1e6:.2f}"],
                    ["κx (1/mm×10⁻³)", f"{strain_curvature[3][0]*1e3:.4f}"],
                    ["κy (1/mm×10⁻³)", f"{strain_curvature[4][0]*1e3:.4f}"],
                    ["κxy (1/mm×10⁻³)", f"{strain_curvature[5][0]*1e3:.4f}"]
                ], columns=["Property", "Value"])
                st.table(midplane_df)
                
                # Display maximum stress locations
                st.subheader("Maximum Stress Locations")
                max_stress_df = pd.DataFrame([
                    ["σx", f"Layer {max_stress_locations['x']['layer']} ({max_stress_locations['x']['surface']})"],
                    ["σy", f"Layer {max_stress_locations['y']['layer']} ({max_stress_locations['y']['surface']})"],
                    ["τxy", f"Layer {max_stress_locations['xy']['layer']} ({max_stress_locations['xy']['surface']})"]
                ], columns=["Stress Component", "Location"])
                st.table(max_stress_df)
                
                # Calculate and display total thickness
                total_thickness = np.sum(given_data[:, 5])
                st.info(f"Total Laminate Thickness: {total_thickness:.2f} mm")
            
            # Tab 5: Layer Properties
            with analysis_tabs[4]:
                st.header("Layer Properties")
                
                layer_tabs = st.tabs([f"Layer {i+1}" for i in range(len(st.session_state.layers))])
                
                for i, tab in enumerate(layer_tabs):
                    with tab:
                        layer = st.session_state.layers[i]
                        mat = st.session_state.materials[layer['material']]
                        
                        # Material info
                        st.subheader("Material Properties")
                        if mat['type'] == "Fiber/Matrix":
                            Vf = float(layer.get('Vf', 0.6))
                            props = mat['props']
                            E1, E2, G12, v12 = micro(
                                props['E1f'], props['E2f'], props['v12f'],
                                props['Em'], props['vm'],
                                float(layer['angle']), Vf
                            )
                            
                            # Show fiber and matrix properties
                            st.markdown(f"**Material Type:** Fiber/Matrix Composite")
                            st.markdown(f"**Fiber Volume Fraction:** {Vf:.2f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("##### Fiber Properties")
                                st.markdown(f"E1f: {props['E1f']:.1f} GPa")
                                st.markdown(f"E2f: {props['E2f']:.1f} GPa")
                                st.markdown(f"ν12f: {props['v12f']:.3f}")
                            
                            with col2:
                                st.markdown("##### Matrix Properties")
                                st.markdown(f"Em: {props['Em']:.1f} GPa")
                                st.markdown(f"νm: {props['vm']:.3f}")
                        else:
                            E1 = mat['props']['E1']
                            E2 = mat['props']['E2']
                            G12 = mat['props']['G12']
                            v12 = mat['props']['v12']
                            st.markdown(f"**Material Type:** Pre-defined Composite")
                        
                        # Calculated ply properties
                        st.subheader("Calculated Ply Properties")
                        st.markdown(f"**Orientation:** {layer['angle']}°")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### Stiffness Properties")
                            st.markdown(f"E1: {E1:.1f} GPa")
                            st.markdown(f"E2: {E2:.1f} GPa")
                            st.markdown(f"G12: {G12:.1f} GPa")
                            st.markdown(f"ν12: {v12:.3f}")
                        
                        with col2:
                            st.markdown("##### Strength Properties")
                            st.markdown(f"Xt: {mat['strength']['Xt']:.1f} MPa")
                            st.markdown(f"Xc: {mat['strength']['Xc']:.1f} MPa")
                            st.markdown(f"Yt: {mat['strength']['Yt']:.1f} MPa")
                            st.markdown(f"Yc: {mat['strength']['Yc']:.1f} MPa")
                            st.markdown(f"S: {mat['strength']['S']:.1f} MPa")
                        
                        # Q-bar matrix
                        st.subheader("Stiffness Matrix (Q-bar)")
                        Q_bar_df = pd.DataFrame(
                            Q_bar_data[i],
                            columns=['1', '2', '6'],
                            index=['1', '2', '6']
                        )
                        st.dataframe(Q_bar_df)
                        
                        # Stresses and strains
                        st.subheader("Layer Results")
                        
                        layer_results_tabs = st.tabs(["Stresses", "Strains", "Failure"])
                        
                        with layer_results_tabs[0]:
                            # Global stresses
                            st.markdown("##### Global Stresses (MPa)")
                            global_stress_df = pd.DataFrame([
                                ["Upper Surface", f"{Layer_stress_upper[i, 0]:.2f}", f"{Layer_stress_upper[i, 1]:.2f}", f"{Layer_stress_upper[i, 2]:.2f}"],
                                ["Middle", f"{Layer_stress_middle[i, 0]:.2f}", f"{Layer_stress_middle[i, 1]:.2f}", f"{Layer_stress_middle[i, 2]:.2f}"],
                                ["Lower Surface", f"{Layer_stress_lower[i, 0]:.2f}", f"{Layer_stress_lower[i, 1]:.2f}", f"{Layer_stress_lower[i, 2]:.2f}"]
                            ], columns=["Location", "σx (MPa)", "σy (MPa)", "τxy (MPa)"])
                            st.dataframe(global_stress_df)
                            
                            # Fiber stresses
                            st.markdown("##### Fiber Stresses (MPa)")
                            fiber_stress_df = pd.DataFrame([
                                ["Upper Surface", f"{fiber_stress_upper[i, 0]:.2f}", f"{fiber_stress_upper[i, 1]:.2f}", f"{fiber_stress_upper[i, 2]:.2f}"],
                                ["Middle", f"{fiber_stress_middle[i, 0]:.2f}", f"{fiber_stress_middle[i, 1]:.2f}", f"{fiber_stress_middle[i, 2]:.2f}"],
                                ["Lower Surface", f"{fiber_stress_lower[i, 0]:.2f}", f"{fiber_stress_lower[i, 1]:.2f}", f"{fiber_stress_lower[i, 2]:.2f}"]
                            ], columns=["Location", "σ1 (MPa)", "σ2 (MPa)", "τ12 (MPa)"])
                            st.dataframe(fiber_stress_df)
                        
                        with layer_results_tabs[1]:
                            # Global strains
                            st.markdown("##### Global Strains (με)")
                            global_strain_df = pd.DataFrame([
                                ["Upper Surface", f"{Layer_strain_upper[i, 0]*1e6:.2f}", f"{Layer_strain_upper[i, 1]*1e6:.2f}", f"{Layer_strain_upper[i, 2]*1e6:.2f}"],
                                ["Middle", f"{Layer_strain_middle[i, 0]*1e6:.2f}", f"{Layer_strain_middle[i, 1]*1e6:.2f}", f"{Layer_strain_middle[i, 2]*1e6:.2f}"],
                                ["Lower Surface", f"{Layer_strain_lower[i, 0]*1e6:.2f}", f"{Layer_strain_lower[i, 1]*1e6:.2f}", f"{Layer_strain_lower[i, 2]*1e6:.2f}"]
                            ], columns=["Location", "εx (με)", "εy (με)", "γxy (με)"])
                            st.dataframe(global_strain_df)
                            
                            # Fiber strains
                            st.markdown("##### Fiber Strains (με)")
                            fiber_strain_df = pd.DataFrame([
                                ["Upper Surface", f"{fiber_strain_upper[i, 0]*1e6:.2f}", f"{fiber_strain_upper[i, 1]*1e6:.2f}", f"{fiber_strain_upper[i, 2]*1e6:.2f}"],
                                ["Middle", f"{fiber_strain_middle[i, 0]*1e6:.2f}", f"{fiber_strain_middle[i, 1]*1e6:.2f}", f"{fiber_strain_middle[i, 2]*1e6:.2f}"],
                                ["Lower Surface", f"{fiber_strain_lower[i, 0]*1e6:.2f}", f"{fiber_strain_lower[i, 1]*1e6:.2f}", f"{fiber_strain_lower[i, 2]*1e6:.2f}"]
                            ], columns=["Location", "ε1 (με)", "ε2 (με)", "γ12 (με)"])
                            st.dataframe(fiber_strain_df)
                        
                        with layer_results_tabs[2]:
                            # Failure analysis
                            strength = strengths[i]
                            
                            # Upper surface
                            st.markdown("##### Upper Surface")
                            fs_upper = fiber_stress_upper[i]
                            f1_upper = fs_upper[0]/strength[0] if fs_upper[0] > 0 else -fs_upper[0]/strength[2]
                            f2_upper = fs_upper[1]/strength[1] if fs_upper[1] > 0 else -fs_upper[1]/strength[3]
                            fs_upper_shear = abs(fs_upper[2])/strength[4]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Fiber Direction", f"{f1_upper:.3f}", 
                                      delta="FAIL" if f1_upper > 1.0 else "SAFE")
                            with col2:
                                st.metric("Matrix Direction", f"{f2_upper:.3f}", 
                                      delta="FAIL" if f2_upper > 1.0 else "SAFE")
                            with col3:
                                st.metric("Shear", f"{fs_upper_shear:.3f}", 
                                      delta="FAIL" if fs_upper_shear > 1.0 else "SAFE")
                            
                            # Lower surface
                            st.markdown("##### Lower Surface")
                            fs_lower = fiber_stress_lower[i]
                            f1_lower = fs_lower[0]/strength[0] if fs_lower[0] > 0 else -fs_lower[0]/strength[2]
                            f2_lower = fs_lower[1]/strength[1] if fs_lower[1] > 0 else -fs_lower[1]/strength[3]
                            fs_lower_shear = abs(fs_lower[2])/strength[4]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Fiber Direction", f"{f1_lower:.3f}", 
                                      delta="FAIL" if f1_lower > 1.0 else "SAFE")
                            with col2:
                                st.metric("Matrix Direction", f"{f2_lower:.3f}", 
                                      delta="FAIL" if f2_lower > 1.0 else "SAFE")
                            with col3:
                                st.metric("Shear", f"{fs_lower_shear:.3f}", 
                                      delta="FAIL" if fs_lower_shear > 1.0 else "SAFE")
                            
                            # Overall status
                            overall_fi_upper = max(f1_upper, f2_upper, fs_upper_shear)
                            overall_fi_lower = max(f1_lower, f2_lower, fs_lower_shear)
                            overall_fi = max(overall_fi_upper, overall_fi_lower)
                            
                            st.markdown("##### Layer Status")
                            st.metric("Overall Safety Factor", f"{1.0/overall_fi:.2f}",
                                  delta="FAIL" if overall_fi > 1.0 else "SAFE")
                            
                            if overall_fi > 1.0:
                                failure_mode = "Fiber Tension" if f1_upper == overall_fi and fs_upper[0] > 0 else (
                                              "Fiber Compression" if f1_upper == overall_fi and fs_upper[0] <= 0 else (
                                              "Matrix Tension" if f2_upper == overall_fi and fs_upper[1] > 0 else (
                                              "Matrix Compression" if f2_upper == overall_fi and fs_upper[1] <= 0 else (
                                              "Shear" if fs_upper_shear == overall_fi else (
                                              "Fiber Tension" if f1_lower == overall_fi and fs_lower[0] > 0 else (
                                              "Fiber Compression" if f1_lower == overall_fi and fs_lower[0] <= 0 else (
                                              "Matrix Tension" if f2_lower == overall_fi and fs_lower[1] > 0 else (
                                              "Matrix Compression" if f2_lower == overall_fi and fs_lower[1] <= 0 else "Shear"
                                              ))))))))
                                
                                st.warning(f"This layer fails due to: **{failure_mode}**")

    # FAQ Section
    with st.expander("❓ FAQ & Help"):
        st.markdown("""
        ## Frequently Asked Questions
        
        ### Analysis Questions
        
        **Q1: How do I find the strain in a specific ply?**  
        A: The strain values are displayed in the "Stresses & Strains" tab. Select the desired visualization type (Global or Fiber strain) and look at the appropriate layer.
        
        **Q2: How do I find the stress in a specific ply?**  
        A: Similar to strains, stress values are in the "Stresses & Strains" tab. For fiber-direction (longitudinal) stresses, select "Stress (Fiber)" and look at the "x/1-direction" component.
        
        **Q3: How do I find values from the ABD matrix?**  
        A: The complete ABD matrix is shown in the "Stiffness Matrix" tab. You can find specific values like D21 there.
        
        **Q4: Where is the maximum stress located?**  
        A: The "Laminate Properties" tab has a "Maximum Stress Locations" section showing which layer and surface (upper/lower) has the maximum stress for each component.
        
        **Q5: How do I find the laminate modulus?**  
        A: The "Laminate Properties" tab displays effective laminate moduli including Ex, Ey, and Gxy.
        
        **Q6: Which layer will fail?**  
        A: The "Failure Analysis" tab shows safety factors and failure indices for each layer. Layers with safety factors < 1.0 or failure indices > 1.0 are predicted to fail.
        
        **Q7: How do I find Q-bar matrix values?**  
        A: The Q-bar matrices for each layer are displayed in the "Stiffness Matrix" tab. Select the specific layer tab to see its Q-bar values.
        
        **Q8: How do I find midplane strains?**  
        A: The "Laminate Properties" tab contains the "Midplane Deformation" section with midplane strains (εx, εy, γxy).
        
        **Q9: How do I find laminate curvatures?**  
        A: The "Laminate Properties" tab shows curvature values (κx, κy, κxy) in the "Midplane Deformation" section.
        
        **Q10: How do I determine the failure mode for a ply?**  
        A: In the "Failure Analysis" tab, each layer's failure modes are listed. You can also go to the "Layer Properties" tab and select a specific layer's "Failure" subtab for detailed failure analysis.
        """)

if __name__ == "__main__":
    main()