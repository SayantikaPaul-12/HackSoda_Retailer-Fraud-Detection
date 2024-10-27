import streamlit as st
import os
import pandas as pd
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Check if the Mistral API key is set in the environment variables
if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = '#use your api key here'

# Initialize the LLM
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

# Load the dataset
@st.cache_resource
def load_data():
    return pd.read_csv("data.csv")

data = load_data()

# Streamlit application layout
st.markdown(
    """
    <style>
        /* Center the title */
        h1 {
            text-align: center;
            color: white;
        }
        
        /* Center the search bar and button */
        .centered-input {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* Search input field styling */
        .stTextInput input {
            font-size: 22px; /* Increase font size */
            padding: 20px;    /* Increase padding */
            width: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            border-radius: 10px;
        }

        /* Search button styling */
        .stButton button {
            font-size: 22px; /* Increase button text size */
            padding: 15px 30px;
            background-color: transparent; /* Transparent by default */
            color: white;
            border-radius: 10px;
            border: 2px solid orange;
            transition: background-color 0.3s, color 0.3s;
            display: block;
            margin: 0 auto;
        }

        /* Search button hover effect */
        .stButton button:hover {
            background-color: orange; /* Orange fill on hover */
            color: black;             /* Black text on hover */
        }

        /* Customize the detail boxes */
        .detail-box {
            padding: 20px;
            margin: 10px;
            background-color: rgba(0, 0, 0, 0.6);
            color: #fff;
            border-radius: 8px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s, box-shadow 0.3s;
            font-size: 18px;
            text-align: center;
        }

        .detail-box-label {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;  /* Adds space between label and value */
            text-align: center;   /* Center the label above the value */
        }

        .detail-box-value {
            font-size: 24px;     /* Larger font for the value */
        }

        .detail-box:hover {
            transform: scale(1.02);
            box-shadow: 2px 2px 12px rgba(255, 255, 255, 0.6);
        }

        /* Color coding for legitimacy */
        /* Color coding for legitimacy */
        .legitimate {
            background-color: #4CAF50 !important; /* Existing background color */
            width: 100%; /* Set the desired width (you can adjust the percentage or use px) */
            padding: 10px; /* Optional: Add some padding for better visual presentation */
            border-radius: 5px; /* Optional: Rounded corners */
        }


        .stTextInput label {
            display: none;
        }

        /* Footer styling */
        .footer {
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px;
            text-align: center;
            margin-top: 40px;
            border-top: 2px solid orange;
        }

        .footer a {
            color: orange;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        /* Logo styling */
        .logo {
            height: 20px;
            vertical-align: middle; /* Align logo with text */
            margin-left: 5px; /* Space between text and logo */
            margin-right: 5px; /* Space between logos */
        }
    </style>
    """, unsafe_allow_html=True)

# Title of the application
st.title("E-Commerce Retailer Fraud Detection")

# Input field for retailer name, centered
with st.container():
    st.markdown("<div class='centered-input'>", unsafe_allow_html=True)
    retailer_name = st.text_input("Retailer Name", placeholder="Search")
    st.markdown("</div>", unsafe_allow_html=True)

# Button to trigger LLM analysis, centered
with st.container():
    st.markdown("<div class='centered-input'>", unsafe_allow_html=True)
    search_button = st.button("Search")
    st.markdown("</div>", unsafe_allow_html=True)

# Function to get retailer details using LLM
def get_retailer_details(retailer_name):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
            """You are provided with a dataset containing details of retailers selling their products on e-commerce platforms.
            Your job is to provide the details of the retailer about whom it is asked and determine whether the given retailer is legitimate or verified or not.
            If the data does not have any information, simply answer as 'No information available'. Do not give further explanation or description.
            Data:{data}
            Retailer:{retailer}
            Present the answer in bullet point format. Please make sure to ultimate judgement whether the retailer is legitimate or not, Also give the address and country of origin and give clean text without any special characters or stylings.""",
            )
        ]
    )
    chain = prompt | llm
    run = chain.invoke(
        {
            "data": data,
            "retailer": retailer_name
        }
    )
    return run.content

# Analyze retailer details
if search_button:
    if retailer_name:
        # Call the function to get retailer details
        with st.spinner('Analyzing retailer details...'):
            try:
                result = get_retailer_details(retailer_name)

                # Display the retailer details in two columns
                st.subheader("Retailer Details")
                details = result.split("\n")
                col1, col2 = st.columns(2)

                # Show alternating details in two columns with labels and values
                for idx, detail in enumerate(details):
                    # Split the detail into label and value
                    if ":" in detail:
                        label, value = detail.split(":", 1)
                        label = label.strip()  # Clean up spaces around label
                        value = value.strip()  # Clean up spaces around value
                    else:
                        label = "Unknown"
                        value = detail.strip()

                    # If legitimacy, color green
                    if "Legitimacy" in label and "Legitimate" in value:
                        box_style = "legitimate"
                    else:
                        box_style = ""

                    # Display the label and value in each box
                    box_content = f"""
                    <div class="detail-box {box_style}">
                        <div class="detail-box-label">{label.replace('-', '').strip()}</div>  <!-- Remove the dash -->
                        <div class="detail-box-value">{value}</div>
                    </div>
                    """

                    # Alternate columns
                    if idx % 2 == 0:
                        with col1:
                            st.markdown(box_content, unsafe_allow_html=True)
                    else:
                        with col2:
                            st.markdown(box_content, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a retailer name to search.")

# Footer
# Footer
# Footer

st.markdown("""
<div class="footer">
    <p>Made for ASU SODA Hacks 2024 by:</p>
    <p>
        <a href="https://www.linkedin.com/in/sutharapuhashwanth/" target="_blank">
            Hashwanth Sutharapu
            <img class="logo" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMQEBUTEhIQFRUVFRUVFhYVFRUVFxgXFRUXFhUVFxUYHSggGBolHRUWITEhJSkrLi4uGB8zODMtNygtLisBCgoKDg0OFxAQGy0fIB4vLS0rMC0wLS0uLy0rLS41NystLTctNS0rKy0tLS0tLS0tLi0tLS0tLS0tLTUrLS0tLf/AABEIAMEBBQMBIgACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAAAQcCAwYEBQj/xABHEAABAgIGBQgHBAoBBQEAAAABAAIDERIhMUFRYQQFgbHwBhMiMkJScaEjM1NicpHBFDSS0QdDY3OCorLS4fEkRFRkg8Kk/8QAGgEBAAIDAQAAAAAAAAAAAAAAAAMFAQIEBv/EAC4RAAIBAwMCBgECBwAAAAAAAAABAgMEEQUSMSFBFSIyUXGBwRORFCMkM2Gx4f/aAAwDAQACEQMRAD8Aum/NN16bsU4GaAYeSb704OScTxQCqWSX53JvwTdigHBTgJwM04OSAX53qLslO7FN+CAb7k4KcTwTgZoBuuS/NOCME3YoBdkm+5N+CcTwQDgpuuTgZpxLBAL80uyvTdim/BAOAnBTg5JwM0AuyuS/NOJJuxQDdenATiWKcHJAN96XZJwM034IBfmm69N2KcSxQDDyTfenByTieKAls7rEWJzMskQE7NicTwTfinBzQDjxTiWCcDJN95xQDbt+ibNmKbsE34oBx4Jx4pwc04GSAcSwzTbt+ib8U3YIBxLFOPBN9xTg5oBxPFNmz6rwa11zA0Vs40VrO6212xorK5bS/wBIjP1MOfvRDL+UfmpYUak/SiGpcUqfSUjuNu36JxLFVjH5dPNsUNGDB+QJ81438smm2LHdtd9Sp1ZVSB31LsW1LjBOJ4qpYfLCF3o44+Je/R+WUO7SIrcnB8txCw7KouxlXtN9yy9mxOJ4LkdA5UOf1YkKL4ET/ls+S+zo2vYbqnAs8ax81BKjOPKJ41oS4Z9XjxTjwUMcCAQQQbCKwp4OajJRxPFNmz6puwTfigHE8E48U3YJwMkA4lgm3b9E4OabsEA2bE4ngm/FODmgHHio4lgnAyTfecUAnlPNFAncZBEBldkm+5L88E3XoBj5puuTgZpvvQCueaXZXpdlim+5AOAmPmnBTgZoBdlclc8033rxa41pC0SC6LGdRht/E51zWi8lZSbeEYbSWWejSdIZCY573NaxtbnOMgNqrblJ+kZzyWaGKLRVzrh0j8DT1fE15Bcxym5TRtYRJv6MJp9HCBqHvO7zs7rl8eiri2sEvNU6v2KS71Bt7afRe5nH0hz3FznOc42ucSSfEmsrS5xWRCxcrJJIq+TAqFJUSQ2E0moRDJm10q19bQOUUeF26bcHzd8nWhfGmgK1lGMujRtGUo9YssnUHK1rjJrjDeew4za7wNh8iu61brdsWTXdF9wuJyP0X5/BXQak5RuhSZFm+Hja5vhiMvlguC4sVLrEsLe/cXiZeV+d6iqWS5/UWvA9rWvcC09SJP5Bx+vzXQ354KnnBxeGXMJqSyhvuTgpuvTgLU2G65L804KXZYoCKpZKT53JXPPBRuvQDHz/AMLEkSyuUm7yz8ViT87xggIeRPpW5IoBPZAIxKIDdxNOJYps2JxPBAOPBOJ4px4pxLBANmxOJpt2ps2YoBx4px4Jx4Jx4oDVpWkMhQ3RIjg1jGlznmwACZmqK5U8o4ms9IpmbYLCRCh4DvuHfd5WePRfpd5RmJEboEI9FtF8eXethwtlTj4twK4yDCoiSt7C3wv1JfRT6ldY/lx+wGrNjJpJfW5MwWv0mE11hiNBzrs22K0k9qb9ijWW0l3Po6q5Ex47A80WNNYpkgkYgAE/NaNdcjI+jsLyGvaLSwkyGJBAMs1baxc2YkRMGohUniVXdnpj2L/wmltxl59/+H5+eyRWJC+nryAGR4jW2NiPaPBriBuXz5K6TyslImYSUELOSghYM5NbgsVtIWMkNshqzBWICkrOTDPsah1wdHdIzMNx6Qw94Z71bvJzWwiNDC4GqbH95srJqiwV0/I/WxY4QiZAmcM4Otltt8fFcV5bqcdy5O6yuXTltfBdXEsU48F5dW6Xz0MOvsORFo+u1erjxVG1joX6eVkcTxTiX1UcSwTbt+iwZG3aoJ/1ioJy2fVQT/vBAQT/AJ91YOOcsDjkhPHe4+q1udlP3e7mgDiL3UDgi1vfK1tP3giA+hvxTgqLslO+5AOBl4pvvTHzTdcgG7BN+KVzzS7K9AODmvBr/WjND0WLpD62wmOeBeSB0W+JMhtXvP8ApVh+nLWtHR4Oig1x4lN49yDIy/G5h/hW9OO6Sj7mspbYtlc6vc+M98eKZviOc9xxc8knZWvoyWvQoVFjRlvW+S9NCO1JI8dXqudRs1kLPRopY4EGRBmCMReoIW7QYFOI1uLgPmZLZ8EZYWqeXEJzAI4c14tLRNpzkKx4LHXHLiG1hEAOLiKnOEg3OVpKkcgYQ/XRPwtUP5Awpeuifhaqf+j3Zy/jrgvM6js24Xz0z/srWPEpGa1UV6jB6UlYMLkDCLQediVgHqtvCsqteFLG7uVdCjUrZVNZwVnJQQrMdyAhe1ifhauf1dyRfGiOANGG1zm0yLZGVQvKjjd0pJtPglnaV4NRcer4OSoLEtVqQuRmjNFfOOOJcB5ALz6ZyJgOHQc9hzk4fKo+ahV/SzjqT+H3CWcL9yslMl9fXWpImjOk8W2OFYIyP0XypLsjJSWUcbym01howWbHEEEGRBmDgRYVBCNW6BbnIfW/OBpNkQScMHt/Ov5hdmf9ZKmOQ+mlr3MnhEb8TSJ/T5K5IMQOaHCxwBO0TVBeUtlQ9HY1f1KayZ8HNYk1ZYXoTZhcoJM/evwlxJch2AznbXjdJayajheMfBCRL3b8ZrBzqx3uzhLNAHGzPqZePktRJmQD0h1jcRgELutL/wBn1l5rS9wkJ9SfQxnn5oDJpcROGQ1uDrc8UXm0h7KXpp076NkrkQH3d+CcHJOJpxLFAOBmm+8Jx4JxPFAN2Kb8E2bE4nggMXmQ35KiP0raWY2t2w59GFBhtHi9znHyLVeOmRJN4rX545RRKeu9IPvsH4YTB9F1WazVRzXbxRkz6LGrMsWcNq3FlSv3I8moZPA4L2amHpmfG3eF54gXp1MPTM+Nu8LMvSyIucqCpKgryp7l8FIv9Yrq0bqN+Fu4KmIjfSK59G6jfhbuCttS4h9lBovqn8L8mbgvOWtY25rRsAnWV6lU/LXXL40dzZmgxxa1t1RkXeJXDbW7rS2p4Ra3l1G3injLfBZMOK1/Vc13wkHchCp7VGsnwYoe0kEH5i8HEK4Ybw5ocLHAOHgRNbXVq6DXXKZrZ3n8RlNYaPn661cNIguhkV2tODhZ+W1VDpEOi6Su0hVHyohhukxQPaO8yT9V1abN+aP2cOq00nCa79GfJKxCFAValWfQ1FHoaTDd7wafB3R+qu/UcWlAbi2bRnI/6VCQHye04EH5FXlyad6J3xmvDotrVbqUeiZbaXLrJH1yfnfl4LW4iVvRuN88N6E/772Swc7KZ7mGfGKqC5DnGdnSubdLFaXOqNfR7RvacApcbqVXtMMuMVqc7KUrGe0zQB7rJ3er9/CflhatReZmQm/ttuaMRwVDnbZ2/sfyls6q0vN1KQFkT2nu54WmxAZsiPA9E0RGXOdKc7xWQi875GsxeYPs5ylnaLUQHTbsE4ngl+abr0A4OacSwTgZJvvQDbtTiSVSyQ+dyA+brd8mniS/PmtTLXMbOIzzhsV+a8PQPmqB5Yej1mH3Oax34SWn+kLptHiqjnuo7qUkdLDatrhUohhbHBXh5pLofPjCtejU/rmfG3eFrjNW7UzfTM+Nu8Ldvys5JrqXGVBUlQV5c9u+CmIg9Irk0fqN+Fu4KnInrFccDqN+Ebla6lxD7PP6H6qnwvybAqS5QfeIn7x/9RV2hUnr/wC8RP3j/wCorGmeqRPrHFP7PnwB0ldOqh/x4X7qH/QFS8C1XXqof8eD+6h/0Bb6n6Ymmkf3J/CNjmqouWH3uL8ZVwEKnuWX3yN8ZUOm+uXwdGq+iHz+D4ihYzQFXJT4NsITIGY3q8eTXqXfFZj0W1Kk9XMpRWD3m75q79RtowGTtcSW5Gcpn5BVupPypFnpi80mfQc7LwHdzK1uNcpyPfuOXGCEmZlaOvmMloe8UZkHm51C+lj4W3qnLolzxKdE0fZ3k48YLW81gTmT1X3Q8jxepeXUpTHOyqd2aOHjbcvOXii4jqA+lF7ji3KcrwgDnW3S6/7b4fGuzvLU9wkCWktPVh3wz3jv2o93UnfLmPcslT/lxsKwm6k4NI50D0ruy5uDc5SuCAwixGtMnwzHd7RoqOAqwRTo7Yjmz0dzWQ65Nf1p33G/NEB1d2Sb7k34JwckAx8/8JuuTgZpvwwQCueai7K9TuxTfggPj68b0dyo79JeiyMOMOw8sd4PrB+bfNXzrWHNp88lVnK7VwjQ4kM9oGifeFbTsIC2hLbJMxJZWD5Wo9I5yAx2Uj4ipfQK4zkVpxa50F9Rmajc5tRHl5Ls16CEtyTPN1aeybieeMxbtTN9Mz4m7wjgsdHic28OwIPyW74aOSUeuS3yoK4c8t4nchfJ39y1xOXMSVTIXyd/cqZWNb2L6WqW69/2OUiesVyQOq34RuVKPi9Ka6xnL2KABzcGoAWPu/iVhe286qjt7FRpl1Tt3Nz74/JYYVKa/wDvET94/wDqK6g8v43s4Pyf/cuO06OYj3PNrnFxlZWZrFjbzpOTl3JdQu6dxs2Z6ZNUDrK7NUj/AI8H91D/AKAqSYZFddonLyNDhtYGQSGNa0Eh05NEhPpZLe+oTqqKj2MWFzChOTn3RY7gqa5Z/fI3xldC79IUb2cD5P8A7lx2t9OOkRXxHAAvcXECcq8JqOytalKTcvYnvbunXjFQz0Z4FIUyRWJwH0+T8IujtkJ4eJqCu2CwMZRFjWhsTKQlV5qrv0d6u5zSaZqZCFNx941Qx86/4VaDnWXS6g9p4+XzVJqM8zUfYutMhim5e7Ie4SE+rP0eJOawLnUjKXOy6Q7NGqzPq3qHPNchMnrtuYMQvO8ijIuIh3Re0T3Tlb+FV5ZkOcygazzM+ke1Sq8rLljEcaTZy5yXoRcWy7Wcp4I97qU6I5yVUKqiR3jdO35LSSJEAksJ9I++Ge63KchtQCl16N/3n3bZ0P58blrdRotpE80D6EjrF2DspzwWR7M6peq/bYU/Ho2y6xUgmZIbN59ZDuhjvNzsO1AatKEIu/5Jc2LeGdWXZuNyhboTntEoUMRmXPdKZN4rwRZB1m3anEsU3YJwMlgDjwTieKcHNN2GCAbNibduCb8U3YIDRpTJjitcDyj0WRJl/hWI4f4yXOcoNCpA+eaAoXlXoTtHjjSGTovIpEXPFh27wcV1WqNPEeEHi2xwwP5L1651e17XQ3tm1wkR9fEWrgdFjRNXaRQfW02G57LiM9xVjZ18eRlfe2+9blyd+QtbmpoukNitDmGYPEjmthCs0ylaPK9i0xGL2kLW9i2TIZQyfPc1ayF7IkNaHNUqZztNGlYlZuCwcFkyjErEqSiybowIWBatsliVg2TNJCMbM2E4AXm4KYjpLtf0e8nqUtMjMmxp9Cy9x9rLui7Ou4KCvWjSi5M6rejKtLajreSuqfsmjNa4ekd0ojO+4iwZNFX8JN6+o522d/sfy8uqpeTMCc3HqvuaMD5/NaC62VQHrB7XGj52Yrzk5OcnJ9z00IKEVFcIOOcpdv23u54XrU599Cf7Du+/KX07SPcJAkEtPq23wji7fWtZpUi0OAiyrjdlze6Lp2fhK1NiHYc5P/ye77k5/XtLEm+jRl+q9v78r8bDYsZtolwYRCvg9ou7wvlZfcsiDMAmbz6uJdCHddnaK8UBHnP/APN+UtnUSV1OjL9d7b3J34WmxB2pVUfW/t7Z0PHpWS6yEiQJaSwn0cO+Ge86+Vp2oBRpV879m/ZTlLOUxbbYixjOY0yiw3Rn3vbORFwqlYiA7C/NN16i7JTvuQDDyTfenBTdcgIuyU77kvzTdegHBXn0uCHN3L0cBJf5QFf6+1YQSQK1xGvNUM0hhY+qVbXC1rsR+SufWGhh7crlxOuNUkEmVayngFPaLpkbV8WhEEwfwvGLTcdy7bV+sGR20mGeIvHiFOs9WsitLIjaTfMHFpuK43S9TaRobqcEuewXjrgZtFuz5Kxt7vtIrrmz3eaJ3RCxIXM6q5WtdIRRI94fUfkuj0fSWRBNjmuGR3i5WKknwVM6covDRD2LREhr2yWDmrZPBBKGT5r2LUQvfFhryvapU8nNKLiectWJatpWiNHa20hZyZjl8Arzx4wbatA0t0V/NwGOiPNjWiZ8chmV3nJX9HwaWxtPk9xkYcESdDndzh7V1Vltq5q13Cmv8ljbWFSp1fRHzORvJQ6VR0jSQ5ujT6DbHRjd4Q6rb7sVagBa4NbREUNkJVMDBUABd8kBIcaIHOSk5vZDapEfy33rzuLKEpnmZ9btUsJSs2KjrV5VZZkX9GjGlHESS9tFxE+aB9IO0XVWZTorW93Unafu+QqlT/lxvUve6mCQOdA9G3subXW7OVK8WLTSHTlYfvHuGudDHtY2BQkxMzSeGypgenNxbfRzl4LS4s5sFwP2efQb2w+uZOXXvvCydKTKRNAEcwRa51wfgJ5BZNL6ZIA+0S6bD1AyqRBnb1L7ygJIfzgBLftEui7sUK6jn1rlraW0XFvqgfTjtF2LcpywUAM5sgOd9nn0n9sPqqAlZ1br1scTSaXS50D0Ley5srXYGU7wgMT+rpXy+zS7Nkqf8mNhWTQ6k4NI50D0x7Jbg3OUsFi3t0b/ALz7ls6GPbxsCghtFocTzQPoXdpzsHZTncEBt0YRS0fZi1sKuQfbOddxvRaNJbDc6ekucyJe1lbZXXG7NEB2O/BOJ4Jt2pxLFAOBmnEsE48E4nigG2rFNmxNmxNu3BAODknAzTjxTjwQEEf6Xi0zQg8fVe7ieOSbNn1QHD611HeBsXNaVq9zTZ/hWxEgg/ngvmaXqlrrv8oCmta8nYMatzKLu+zouPjcdoXPxuTOkQjODED5XE0HfkfmFdOl8n8BswXyo2oSPzUkK04cM0lTjLlFVDWmnQevDikZtpj8QnvWTeWbhU5rJ7Qd6sl2pnYbFgdSE2ifiJyXTG+muUc0rGkyuncsB3W/MrUOUT4nUYXfC1zt01Z8Hk6O40fwjpeC+po2opXfw93Nb+IT7Ij8Opdyp9H0HT9IMmQXtne+TB8jX5Lo9T/o3dEIdpUclt4hVAZF7qzdYArL0bVbW1TA/aXeE/8AK9rWACdGX7O93vS/xcoJ3dWfc6KdrShwj5mpdRwdEYGwYTIeAAri5uca3HxX0ibZCc+sPZZjz+SOOc52O9lkcPKxaXOzlK0+2yGM9tq5m8nQHkSkXENui3vPdPn+Fa3PM50BTl6iqjLvYTUOffRpA/qZVw/fIu+XaWt2HOCf/c3fBOf1QEOIkQHEsPWi3wz3RlYP4liTZOqXqx7fAuxnVb3lE76MgLYEq4vvgX42HqptnOw/9tkcJbOqgJBMyQJuPXZdBHebhjUokKMi8hl0efSce4TbK38KmV1KUrYnt/dBvwtKidU6ExZ9mlWz9pKX07aAypGc6AESVUCqi4d+Vk7fksRKRAJLSenEvhHutwF1WKmV3OTP/c3N9yc/r2knfRogWwpev98C/Gw2IBhOqXq//IwpYzqt7yAmZIaC49eFdCHeG/aowvnZ/wCN44S2dRSBdSkRbGuje4DfhabEBLHOAlDhCO32jpEnEV4WKEDSaxFGjj2RqlnKYttsRAdbuwTgZJXPNN16AcHNN1wTDyTfegG/FN2CiqWSm/O5AOBknBzTgpwEA3YJvxS/O9RVLJATuvCg/wCslJ87lGPmgMSyf1zWl+jtIsqwW8n5XLEkz96/CSA8rtDE7p43LX9kbhVeMfBeouEvdvxnxJYOJmO92fDNAaOYAlVb1fd8fJKNZAqcOsbiMAsi7rS/9n1l5rS9wkJ9SfQxnn5oCHObRnI833O1PHgrB5NICY5w9V/ZAwOdt16lxdTkJc9L+GjxJeVzm0DKfMz6fepVSllOigMnOEnSEmj1oviHFvnhatT3CTZiYMuaF8M3F3lbOxTEcZsnKmfUYSupbJLVMzfRlS/6jCVc6GykgHSpEBwEUCb4nZc2roi6dbbh1StVJlClQdzE5c126XenOctqhxZQbSnzE/Rd6nXOeU6fktvpOc7P2mX8FD80BBDqQBIMU1w4nZY2XVddOVK42hYjtSEg3149qa5lmHaslaFDaNB1GfMT9LPrU6urlOj5rJ1sOlKf/TeFUqf8nmgIJEmkglhI5lt8M4uynjNZBrqZaHARpTdF7Lm1dECydbbuyUbOk+jLnJf8jCj7uclrdQ5sTn9nn0O/TrnPLr+SABzKBcGuEGcjC7Zd3gZzlZfcsyDSaCZxD6p46sNsuq7OU7QbVkec5wTo/aZdHuUK7c+stbaNF9H1U/T96l7uU5LIJHalVR9f+1tnQwn0rJdYKCW0WktJhk+iZ2obu87Kc7zapP6ull9m8pU/5PNZNpU3UZc7L03do+7nKSwDXHdDaZR2OivveyYBFwqIs8EW7Redoj7NR5quVO2c+l5ogOquS8KUQEC9DYERATeoFiIgGCnFEQEGxTeiIDEWFQ65EQEXlan9UKUQEO648FoPVd4oiA1RP1ez6LEesf4fkpRAeKJ6n+L81nF9ez4fo5EQHkb1I3id5WuJ1YHiPopRAbIfr4nw/wBq8v8A038SIgPVE9fD+E7nLVDsj+J/+lKIDGJ6uD8Q3rfD+8v+D+1SiA8bfux+L8l6YnrYPw/QoiAxh/r9v/0tcT1ML4vqURAatbetPgERFkH/2Q==" alt="LinkedIn Logo">
        </a>, 
        <a href="https://www.linkedin.com/in/sayantikapaul12" target="_blank">
            Sayantika Paul
            <img class="logo" src="data:image/jpeg;base64,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" alt="LinkedIn Logo">
        </a>, 
        <a href="https://www.linkedin.com/in/ojas-deodhar-131b39197/" target="_blank">
            Ojas Deodhar
            <img class="logo" src="data:image/jpeg;base64,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" alt="LinkedIn Logo">
        </a>
    </p>
    <p>
        <img class="logo" src="https://png.pngtree.com/png-clipart/20220521/ourmid/pngtree-red-location-icon-sign-png-image_4644037.png" alt="Location Logo">
        Tempe, AZ
    </p>
</div>
""", unsafe_allow_html=True)

