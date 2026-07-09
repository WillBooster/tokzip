import { enUsModule } from '../generated/enUs.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(enUsModule);

export { enUsModule };
