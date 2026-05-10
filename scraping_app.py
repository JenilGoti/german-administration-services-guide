from scrapping.flush_all_situations_mapping import main as flush_all_situations_mapping
from scrapping.flush_services_from_listing import main as flush_services_from_listing
from scrapping.flush_sub_situations_from_listing import main as flush_sub_situations_from_listing


if __name__ == "__main__":
    print("Flushing situations")
    flush_all_situations_mapping()
    print("Flushing sub situations")
    flush_sub_situations_from_listing()
    print("Flushing services")
    flush_services_from_listing()
    

